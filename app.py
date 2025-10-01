import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = joblib.load('models/kmeans_materials.pkl')
kmeans = data['model']
cluster_names = data['cluster_names']

pca = joblib.load('models/pca_transformer.pkl')

st.title("Materials Classification Project")

st.markdown("""
This project is designed to help engineers, researchers, and students quickly classify materials based on their mechanical properties.  
By analyzing key properties such as:

- **Ultimate Tensile Strength (UTS):** The maximum stress that a material can withstand while being stretched or pulled before breaking.  
- **Yield Strength:** The stress at which a material begins to deform plastically.  
- **Elongation at Break (%):** How much a material can stretch before it breaks, indicating ductility.

the system groups materials into clusters representing distinct material types.  

The main objective is to **simplify material selection** for different engineering applications. Instead of manually comparing tables and charts, users can input material properties through the sidebar, and the app will provide:  

1. **The predicted material type** based on clustering.  
2. **A detailed description** of the material type, highlighting its mechanical behavior and suitable applications.  

This approach can save time in research, design, and educational projects by giving quick insights into material performance and usability.
""")

st.sidebar.header("Enter Material Properties")

uts = st.sidebar.number_input("Ultimate Tensile Strength", value=572.0)
ys = st.sidebar.number_input("Yield Strength", value=387.0)
elong = st.sidebar.number_input("Elongation at Break", value=18.0)
bh = st.sidebar.number_input("Brinell Hardness Number", value=172.0)
emod = st.sidebar.number_input("Elastic Modulus", value=164571.0)
smod = st.sidebar.number_input("Shear Modulus", value=85598.0)
pratio = st.sidebar.number_input("Poisson's Ratio", value=0.3)
density = st.sidebar.number_input("Density", value=6929.85)



sample = np.array([[uts, ys, elong, bh, emod, smod, pratio, density]])
sample_pca = pca.transform(sample)
cluster = kmeans.predict(sample_pca)[0]
cluster_name = cluster_names[cluster]

st.subheader("Prediction Result")
st.write(f"**Cluster {cluster}: {cluster_name}**")



X_pca_all = joblib.load('models/X_pca_all.pkl')
kmeans_labels_all = joblib.load('models/kmeans_labels_all.pkl')



# Create a scatter plot of the PCA data, colored by the K-Means labels
X_pca_all = joblib.load("models/X_pca_all.pkl")
kmeans_labels_all = joblib.load("models/kmeans_labels_all.pkl")
# Transform original cluster centers into PCA space
cluster_centers_pca = kmeans.cluster_centers_


plt.figure(figsize=(10, 8))


scatter = plt.scatter(X_pca_all[:, 0], X_pca_all[:, 1], 
                      c=kmeans_labels_all, cmap='viridis', s=50, alpha=0.8)

plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
            c='red', marker='X', s=200, label='Centroids')


plt.scatter(sample_pca[0, 0], sample_pca[0, 1], 
            c='black', marker='X', s=200, label='Your Input')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering on PCA-Reduced Data")
plt.legend()
plt.colorbar(scatter, label="Cluster Label")

st.pyplot(plt)

#function to get cluster description
def get_material_description(cluster):
    descriptions = {
        0: """
**Cluster 0: Low-Strength, Brittle Materials**  
These materials have relatively low tensile and yield strengths, and limited elongation. They are typically **brittle**, breaking without much deformation.  
**Applications:** Non-structural components, decorative items, low-load mechanical parts.  
**Examples:** Some ceramics, low-grade plastics, and certain cast irons.
""",
        1: """
**Cluster 1: Medium-Strength, Moderately Ductile Materials**  
Materials in this cluster show a balance between strength and ductility. They can withstand moderate loads and deform slightly before failure.  
**Applications:** General engineering purposes, moderate-load machinery parts, automotive components, and construction materials where flexibility is advantageous.  
**Examples:** Mild steel, aluminum alloys, some polymers.
""",
        2: """
**Cluster 2: High-Strength, Highly Ductile Materials**  
These materials have high tensile and yield strengths and significant elongation at break, making them **strong yet flexible**. They are ideal for demanding mechanical and structural applications.  
**Applications:** Heavy-duty structural components, aerospace, automotive safety parts, and high-stress mechanical systems.  
**Examples:** High-strength steel alloys, titanium alloys, advanced composites.
"""
    }
    return descriptions.get(cluster, "Unknown material type")



# Display result
st.subheader("Material Classification Result")
st.write(f"The material has been classified into cluster **{cluster}**.")
st.write(get_material_description(cluster))



# Plot cluster centers
plt.scatter(
    cluster_centers_pca[:, 0], cluster_centers_pca[:, 1],
    c="black", marker="D", s=200, label="Cluster Centers"
)

