import pickle
#pickle.dump(kmeans,open('unsupervisedmodels.pkl','wb'))
import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('kmeans_mod.pkl','rb')) 
def predict_cluster(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,ContainmentHealthIndexForDisplay,GovernmentResponseIndexForDisplay,EconomicSupportIndexForDisplay,StringencyLegacyIndex,StringencyIndex,ContainmentHealthIndex,EconomicSupportIndex,ConfirmedCases,ConfirmedDeaths):
    input=np.array([[CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,ContainmentHealthIndexForDisplay,GovernmentResponseIndexForDisplay,EconomicSupportIndexForDisplay,StringencyLegacyIndex,StringencyIndex,ContainmentHealthIndex,EconomicSupportIndex,ConfirmedCases,ConfirmedDeaths]]).astype(np.float64)
    prediction=model.predict(input)
    return prediction
 
st.title("CLUSTERING COUNTRIES'RECORDS")
html_temp = """
<div style="background-color:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;">Clustering App </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
CountryName = st.text_input("CountryName")
StringencyLegacyIndexForDisplay = st.text_input("StringencyLegacyIndexForDisplay",)
StringencyIndexForDisplay = st.text_input("StringencyIndexForDisplay")
ContainmentHealthIndexForDisplay = st.text_input("ContainmentHealthIndexForDisplay")
GovernmentResponseIndexForDisplay = st.text_input("GovernmentResponseIndexForDisplay")
EconomicSupportIndexForDisplay = st.text_input("EconomicSupportIndexForDisplay")
StringencyLegacyIndex = st.text_input("StringencyLegacyIndex")
StringencyIndex = st.text_input("StringencyIndex")
ContainmentHealthIndex = st.text_input("ContainmentHealthIndex")
EconomicSupportIndex = st.text_input("EconomicSupportIndex")
ConfirmedCases = st.text_input("ConfirmedCases")
ConfirmedDeaths = st.text_input("ConfirmedDeaths")


 
safe_html="""  
  <div style="background-color:#F4D03F;padding:10px >
   <h2 style="color:white;text-align:center;">In cluster zero</h2>
   </div>
"""
danger_html="""  
  <div style="background-color:#F08080;padding:10px >
   <h2 style="color:black ;text-align:center;"> Cluster one</h2>
   </div>
"""
 
if st.button("Predict"):
  output=predict_cluster(CountryName,StringencyLegacyIndexForDisplay,StringencyIndexForDisplay,ContainmentHealthIndexForDisplay,GovernmentResponseIndexForDisplay,EconomicSupportIndexForDisplay,StringencyLegacyIndex,StringencyIndex,ContainmentHealthIndex,EconomicSupportIndex,ConfirmedCases,ConfirmedDeaths)
  st.success('This country located in cluster {}'.format(output))
