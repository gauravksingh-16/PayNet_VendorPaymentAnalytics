import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from joblib import load as joblib_load

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PayNet QuickPredictor", layout="centered")

# ─── 0) Load model & artifacts ────────────────────────────────────────────────
custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'mae': tf.keras.metrics.MeanAbsoluteError()
}
joint_model = load_model('models/paynet_model.h5', custom_objects=custom_objects)
scaler      = joblib_load('models/scaler.pkl')
ae_model    = Model(
    inputs=joint_model.input,
    outputs=joint_model.get_layer('reconstruction').output
)
threshold = 0.02  # 95th‑percentile for anomaly flag

# ─── 0.5) Load your vendor stats (joblib dump) ────────────────────────────────
vendor_stats = joblib_load('models/vendor_data.pkl')

# ─── 1) Feature defs & scaler lookups ────────────────────────────────────────
features_input = [
    'DaysLate','DaysSinceLastInvoice','DiscountAmount','EarlyPaymentDays',
    'InvoiceAmountRatio','InvoiceDayOfWeek','InvoiceMonth','InvoiceWeek',
    'Monthly_Invoice_Volume','PaymentAmount','PaymentSpeedRatio',
    'PaymentTermDays','ProcessingTime','Vendor_Avg_DaysLate','Vendor_Std_DaysLate'
]
scale_cols = [
    'PaymentAmount','DaysLate','ProcessingTime','PaymentSpeedRatio',
    'PaymentTermDays','EarlyPaymentDays','InvoiceAmountRatio',
    'DiscountAmount','DaysSinceLastInvoice'
]
means  = dict(zip(scale_cols, scaler.mean_))
scales = dict(zip(scale_cols, scaler.scale_))

# ─── 2) Inference with inverse-scaling ─────────────────────────────────────────
def infer_invoice(invoice):
    # a) compute date deltas
    inv_dt = pd.to_datetime(invoice['InvoiceDate'])
    due_dt = pd.to_datetime(invoice['DueDate'])
    pay_dt = pd.to_datetime(invoice.get('PaymentDate', invoice['DueDate']))
    term_days   = (due_dt - inv_dt).days
    proc_days   = (pay_dt - inv_dt).days
    early_days  = max((due_dt - pay_dt).days, 0)
    speed_ratio = proc_days / term_days if term_days > 0 else 0.0

    # b) lookup vendor stats
    v = vendor_stats.loc[vendor_stats['VendorName'] == invoice['VendorName']].iloc[0]

    amt         = invoice['PaymentAmount']
    disc_amt    = amt * invoice.get('DiscountPercentage', 0) / 100
    inv_amt_rat = amt / v['Vendor_Avg_PaymentAmount'] if v['Vendor_Avg_PaymentAmount'] > 0 else 1.0

    mon        = inv_dt.month
    dow        = inv_dt.dayofweek
    week       = inv_dt.isocalendar().week
    days_since = invoice.get('DaysSinceLastInvoice', 0)

    # c) build feature row
    raw = {
        'DaysLate':               0.0,     # placeholder (we’ll scale it)
        'DaysSinceLastInvoice':   days_since,
        'DiscountAmount':         disc_amt,
        'EarlyPaymentDays':       early_days,
        'InvoiceAmountRatio':     inv_amt_rat,
        'InvoiceDayOfWeek':       dow,
        'InvoiceMonth':           mon,
        'InvoiceWeek':            week,
        'Monthly_Invoice_Volume': v['Monthly_Invoice_Volume'],
        'PaymentAmount':          amt,
        'PaymentSpeedRatio':      speed_ratio,
        'PaymentTermDays':        term_days,
        'ProcessingTime':         proc_days,
        'Vendor_Avg_DaysLate':    v['Vendor_Avg_DaysLate'],
        'Vendor_Std_DaysLate':    v['Vendor_Std_DaysLate']
    }
    df = pd.DataFrame([raw], columns=features_input)

    # d) manual scaling of features
    for c in scale_cols:
        df[c] = (df[c] - means[c]) / scales[c]

    # e) predict scaled days‑late + recon
    y_pred_scaled, _ = joint_model.predict(df.values)
    pred_scaled = float(y_pred_scaled.flatten()[0])

    # f) inverse‑scale to get actual days late
    pred_days_late = pred_scaled * scales['DaysLate'] + means['DaysLate']

    # g) anomaly detection on scaled features
    df['DaysLate'] = pred_scaled
    recon = ae_model.predict(df.values)
    mse   = np.mean((df.values - recon) ** 2)
    is_anom = mse > threshold

    return pred_days_late, bool(is_anom)

# ─── 3) Streamlit UI ─────────────────────────────────────────────────────────
st.title("PayNet QuickPredictor")
st.markdown("Enter your invoice details to see predicted days late and an anomaly flag.")

# ---- Layout: two columns side by side on one page ----
col1, col2 = st.columns(2)

with col1:
    invoice_date = st.date_input(
        "Invoice Date", 
        value=pd.to_datetime("2025-03-10")
    )
    due_date = st.date_input(
        "Due Date", 
        value=pd.to_datetime("2025-04-30")
    )
    amount = st.number_input(
        "Payment Amount", 
        min_value=0.0, 
        value=2000.0, 
        step=100.0
    )

with col2:
    discount = st.slider(
        "Discount %", 
        0.0, 100.0, 0.0, 
        step=0.5,
    )
    vendor_name = st.selectbox(
        "Vendor Name", 
        vendor_stats['VendorName'].sort_values().unique()
    )
    with st.expander("Advanced Options"):
        specify_pay = st.checkbox("Specify Payment Date")
        if specify_pay:
            payment_date = st.date_input("Payment Date", value=due_date)
        else:
            payment_date = None

        specify_since = st.checkbox("Specify Days Since Last Invoice")
        if specify_since:
            days_since = st.number_input(
                "Days Since Last Invoice", 
                min_value=0, 
                value=0, 
                step=1
            )
        else:
            days_since = None

# ---- Predict button at bottom of single page ----
if st.button("Predict"):
    payload = {
        'InvoiceDate':        str(invoice_date),
        'DueDate':            str(due_date),
        'PaymentAmount':      amount,
        'DiscountPercentage': discount,
        'VendorName':         vendor_name
    }
    if payment_date:
        payload['PaymentDate'] = str(payment_date)
    if days_since is not None:
        payload['DaysSinceLastInvoice'] = days_since

    days_late, flag = infer_invoice(payload)
    st.metric("Predicted Days Late", f"{days_late:.2f} days")
    if flag:
        st.error("Anomaly Detected!")
    else:
        st.success("All Good!")
