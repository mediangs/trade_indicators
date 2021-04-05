import streamlit as st
from multiapp import MultiApp
import maturity_spreads, trading_indicators # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Maturity Spread", maturity_spreads.app)
app.add_app("Trading Indicators", trading_indicators.app)

# The main app
app.run()
