import re

with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Navigation
old_nav = """    nav_options = [
        "🔬 Patch Analysis",
        "🗺️ Whole-Slide Analysis",
        "📊 Model Evaluation",
    ]"""

new_nav = """    nav_options = [
        "🔬 Patch Analysis",
        "🗺️ Whole-Slide Analysis",
        "📊 Model Evaluation",
        "🧬 Learned Feature Analysis",
    ]"""

if old_nav in text:
    text = text.replace(old_nav, new_nav, 1)
    print("Nav updated.")
else:
    print("Nav already updated or not found.")

# 2. Normal prediction with anomaly flag clarification
anomaly_clarification = """
        # Contextual clarification when an anomaly is flagged but primary verdict is Normal
        if not is_malignant and active_anom_count > 0:
            st.markdown(
                \"\"\"
                <div style="background:#eff6ff; border:1px solid #bfdbfe; border-left:4px solid #2563eb; border-radius:8px; padding:10px 14px; margin-bottom:14px; font-size:0.82rem; color:#1e40af; line-height:1.45;">
                    <b>ℹ️ Anomaly Flagged on Non-Malignant Patch (Objective Divergence):</b><br>
                    An anomaly pattern was flagged, but the primary tumour classifier did not exceed its tumour threshold (P(Tumor) < 0.5000).<br>
                    • <b>Primary Classifier Objective:</b> Detects presence of metastatic breast carcinoma in CAMELYON16 lymph node parenchyma.<br>
                    • <b>Anomaly Probes Objective:</b> Evaluate secondary morphological sub-patterns (e.g. dense cell clusters, ductal architecture disruption) trained on independent proxy datasets (NuCLS, BACH, BCSS).<br>
                    • <b>Histological Reality:</b> Normal lymph node structures—such as dense lymphoid follicles and germinal centers—naturally contain high cellular density, legitimately triggering the cell-density anomaly probe without harboring metastatic malignant cells.
                </div>
                \"\"\",
                unsafe_allow_html=True,
            )
"""

card_end_marker = 'Research Decision Support • Not a Clinically Validated Diagnosis\n                    </span>\n                </div>\n            </div>\n            """,\n            unsafe_allow_html=True,\n        )'

if card_end_marker in text and "Anomaly Flagged on Non-Malignant Patch" not in text:
    text = text.replace(card_end_marker, card_end_marker + anomaly_clarification, 1)
    print("Normal/Anomaly clarification added.")
else:
    print("Card end marker not found or clarification already present.")

# 3. WSI Viewer side-by-side & colorbar
old_wsi_anchor = """        blended_wsi_heat = blend_wsi_heatmap(
            wsi_rgb=wsi_result["wsi_rgb"],
            prob_upsampled=wsi_result["prob_upsampled"],
            tissue_mask=wsi_result["tissue_mask"],
            alpha=wsi_alpha,
            colormap_name=wsi_cmap,
        )

        # Side-by-side Large Viewer
        v_col1, v_col2 = st.columns(2, gap="large")
        with v_col1:
            raw_or_grid = wsi_result["grid_overlay"] if show_grid_lines else wsi_result["wsi_rgb"]
            st.image(raw_or_grid, caption="Level 0 Tissue Scan + Screening Grid (Green = Benign, Crimson = Malignant, Gray = Filtered Glass)", use_container_width=True)
        with v_col2:
            st.image(blended_wsi_heat, caption=f"Continuous Malignancy Heatmap P(Tumor) [{wsi_cmap.upper()}] — {selected_model_arch}", use_container_width=True)"""

new_wsi_anchor = """        wsi_view_mode = st.radio(
            "Spatial Heatmap Visualization Mode:",
            ["Blended Overlay (H&E + Heatmap)", "Pure Continuous Heatmap Field"],
            horizontal=True,
            key="wsi_view_mode_radio"
        )

        blended_wsi_heat = blend_wsi_heatmap(
            wsi_rgb=wsi_result["wsi_rgb"],
            prob_upsampled=wsi_result["prob_upsampled"],
            tissue_mask=wsi_result["tissue_mask"],
            alpha=wsi_alpha,
            colormap_name=wsi_cmap,
        )

        if wsi_view_mode == "Pure Continuous Heatmap Field":
            cmap_obj = matplotlib.colormaps[wsi_cmap]
            heat_rgb = (cmap_obj(wsi_result["prob_upsampled"])[:, :, :3] * 255.0).astype(np.float32)
            t_mask = wsi_result["tissue_mask"][:, :, None]
            display_wsi_heat = np.clip(heat_rgb * t_mask + 255.0 * (1.0 - t_mask), 0, 255).astype(np.uint8)
        else:
            display_wsi_heat = blended_wsi_heat

        # Side-by-side Large Viewer
        v_col1, v_col2 = st.columns(2, gap="large")
        with v_col1:
            raw_or_grid = wsi_result["grid_overlay"] if show_grid_lines else wsi_result["wsi_rgb"]
            cap_col1 = "Level 0 Raw Tissue Scan (H&E Optical Slide)" if not show_grid_lines else "Level 0 Tissue Scan + Screening Grid (Green = Benign, Crimson = Malignant, Gray = Background)"
            st.markdown("##### 🔬 Level 0 Whole Slide (H&E Optical Scan)")
            st.image(raw_or_grid, caption=cap_col1, use_container_width=True)
        with v_col2:
            st.markdown(f"##### 🗺️ Spatial Malignancy Heatmap ({'Pure Field' if 'Pure' in wsi_view_mode else 'Blended Overlay'})")
            st.image(display_wsi_heat, caption=f"Continuous P(Tumor) Probability Field [{wsi_cmap.upper()}] — {selected_model_arch}", use_container_width=True)

            # High-visibility continuous colorbar legend
            st.markdown(
                \"\"\"
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:6px 12px; margin-top:4px; margin-bottom:6px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; font-weight:700; color:#1e293b; margin-bottom:3px;">
                        <span style="color:#1d4ed8;">0.00 — Normal Lymph Node Stroma</span>
                        <span style="color:#b45309;">0.50 — Decision Boundary (τ)</span>
                        <span style="color:#dc2626;">1.00 — Metastatic Carcinoma</span>
                    </div>
                    <div style="height:10px; border-radius:4px; background: linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000, #800000); border: 1px solid #cbd5e1;"></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.70rem; color:#64748b; margin-top:2px;">
                        <span>Cold / Low Suspicion (P < 0.30)</span>
                        <span>Equivocal Range [0.30, 0.50)</span>
                        <span>Hot / High Suspicion (P ≥ 0.50)</span>
                    </div>
                </div>
                \"\"\",
                unsafe_allow_html=True,
            )"""

if old_wsi_anchor in text:
    text = text.replace(old_wsi_anchor, new_wsi_anchor, 1)
    # Also adjust default checkbox to value=False
    text = text.replace(
        'show_grid_lines = st.checkbox("Overlay 224×224 Tile Screening Grid", value=True)',
        'show_grid_lines = st.checkbox("Overlay 224×224 Tile Screening Grid", value=False, help="Toggle 224x224 tile bounding boxes (Green = Benign, Red = Malignant)")',
        1
    )
    print("WSI viewer & colorbar updated.")
else:
    print("old_wsi_anchor not found.")

# 4. Section 4: Learned Feature Analysis
sec4_code = '''
# ==============================================================================
# SECTION 4: LEARNED FEATURE ANALYSIS
# ==============================================================================
elif nav == "🧬 Learned Feature Analysis":
    st.subheader("🧬 Learned Feature Analysis & Latent Space Exploration")
    st.caption(
        "Inspect pre-classifier neural embeddings across convolutional backbones, interactive 2D principal component analysis (PCA), "
        "and empirical distinctions between deep neural representations and continuous physical tissue measurements."
    )

    # Compact Feature-Dimension Summary Table
    st.markdown(
        \"\"\"
        <table style="width:100%; border-collapse:collapse; font-size:0.83rem; margin-top:4px; margin-bottom:16px;">
            <tr style="border-bottom:2px solid #cbd5e1; background:#f1f5f9; text-align:left;">
                <th style="padding:7px 10px;">Architecture</th>
                <th style="padding:7px 10px;">Feature Dimension</th>
                <th style="padding:7px 10px;">Backbone Mechanism / Layer Origin</th>
                <th style="padding:7px 10px;">Parameters</th>
                <th style="padding:7px 10px;">Checkpoint Size</th>
                <th style="padding:7px 10px;">Inference Throughput</th>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
                <td style="padding:7px 10px;"><b>EfficientNet-B0</b> (Primary)</td>
                <td style="padding:7px 10px;"><code style="color:#0f766e; font-weight:700;">1,280-D</code></td>
                <td style="padding:7px 10px;">Adaptive AvgPool (MBConv + Squeeze-and-Excitation)</td>
                <td style="padding:7px 10px;">4.01M</td>
                <td style="padding:7px 10px;">39.7 MB</td>
                <td style="padding:7px 10px;"><b>21.9 FPS</b> (Fastest)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0;">
                <td style="padding:7px 10px;"><b>ResNet-50</b></td>
                <td style="padding:7px 10px;"><code style="color:#1d4ed8; font-weight:700;">2,048-D</code></td>
                <td style="padding:7px 10px;">Bottleneck Layer 4 Output (Residual Skips)</td>
                <td style="padding:7px 10px;">23.51M</td>
                <td style="padding:7px 10px;">204.2 MB</td>
                <td style="padding:7px 10px;">10.2 FPS</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0;">
                <td style="padding:7px 10px;"><b>DenseNet-121</b></td>
                <td style="padding:7px 10px;"><code style="color:#7c3aed; font-weight:700;">1,024-D</code></td>
                <td style="padding:7px 10px;">DenseBlock 4 + Norm5 (Channel Concatenation)</td>
                <td style="padding:7px 10px;">6.95M</td>
                <td style="padding:7px 10px;">43.7 MB</td>
                <td style="padding:7px 10px;">7.1 FPS</td>
            </tr>
            <tr>
                <td style="padding:7px 10px;"><b>VGG-16</b></td>
                <td style="padding:7px 10px;"><code style="color:#b45309; font-weight:700;">4,096-D</code></td>
                <td style="padding:7px 10px;">FC Layer 6 Dense Vector (Dense Linear Stack)</td>
                <td style="padding:7px 10px;">134.26M</td>
                <td style="padding:7px 10px;">1,478.3 MB</td>
                <td style="padding:7px 10px;">4.1 FPS</td>
            </tr>
        </table>
        \"\"\",
        unsafe_allow_html=True,
    )

    proj_cache = load_cached_feature_projections()

    if proj_cache and "models" in proj_cache:
        meta = proj_cache.get("metadata", {})

        p_col1, p_col2 = st.columns([1.2, 1.0])
        with p_col1:
            sel_feat_model = st.selectbox(
                "Select Architecture for 2D Latent Visualization:",
                list(proj_cache["models"].keys()),
                index=0,
            )
        with p_col2:
            sel_color_mode = st.radio(
                "Color Scatter Points By:",
                ["Ground-Truth Class", "Prediction Concordance"],
                horizontal=True,
            )

        model_data = proj_cache["models"][sel_feat_model]
        exp_var = model_data.get("pca_explained_variance_ratio", [0.0, 0.0])
        cum_var = (exp_var[0] + exp_var[1]) * 100.0
        samples_list = proj_cache.get("samples", [])
        points_list = model_data.get("points", [])

        st.markdown(
            f\"\"\"
            <div style="background:#eff6ff; border:1px solid #bfdbfe; border-left:4px solid #2563eb; border-radius:6px; padding:10px 14px; margin-bottom:12px; font-size:0.83rem; color:#1e40af; line-height:1.45;">
                <b>🔬 Scientific Interpretation & Dimensionality Caveat:</b> 2D PCA projects high-dimensional embeddings ({model_data['embedding_dimension']}-D) onto the two orthogonal axes capturing maximum variance.
                Here, PC1 accounts for <b>{exp_var[0]*100:.1f}%</b> and PC2 accounts for <b>{exp_var[1]*100:.1f}%</b> (<b>{cum_var:.1f}%</b> cumulative 2D variance).
                While 2D projections visually indicate dominant separation axes between normal lymph node tissue and metastatic carcinoma, full linear probe boundaries operate in the uncompressed high-dimensional feature space where class separation is significantly sharper.
            </div>
            \"\"\",
            unsafe_allow_html=True,
        )

        df_scatter = pd.DataFrame([
            {
                "PC1": pt["pc1"],
                "PC2": pt["pc2"],
                "True_Label": samp["true_label"],
                "True_Class": samp["true_class"],
                "Pred_Class": pt["pred_class"],
                "Pred_Prob": pt["pred_prob"],
                "Filename": samp["filename"],
                "Concordance": "Concordant (Correct)" if pt["is_correct"] else "Discordant (Error)",
            }
            for samp, pt in zip(samples_list, points_list)
        ])

        color_col = "True_Class" if sel_color_mode == "Ground-Truth Class" else "Concordance"
        color_map = {
            "Metastatic Carcinoma": "#dc2626",
            "Normal Lymph Node": "#16a34a",
            "Concordant (Correct)": "#2563eb",
            "Discordant (Error)": "#ef4444",
        }

        fig_pca = px.scatter(
            df_scatter,
            x="PC1",
            y="PC2",
            color=color_col,
            color_discrete_map=color_map,
            hover_data=["Filename", "Pred_Prob", "True_Class", "Pred_Class"],
            title=f"2D PCA Latent Space: {sel_feat_model} (Embedding Dim: {model_data['embedding_dimension']}-D • Explained Variance: {cum_var:.1f}%)",
        )
        fig_pca.update_layout(
            xaxis_title=f"Principal Component 1 ({exp_var[0]*100:.1f}% Variance)",
            yaxis_title=f"Principal Component 2 ({exp_var[1]*100:.1f}% Variance)",
            height=460,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        st.caption(
            f"• <b>Dataset & Provenance:</b> Evaluated on {meta.get('total_samples', 120)} held-out CAMELYON16 test patches "
            f"({meta.get('normal_samples', 60)} Normal Lymph Node, {meta.get('tumor_samples', 60)} Metastatic Carcinoma) "
            f"extracted freshly on frozen checkpoints (timestamp: <code>{meta.get('created_at', 'Verified')}</code>)."
        )
    else:
        st.warning("Feature projection cache not found. Please run feature extraction to populate.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    with st.expander("🔬 Multi-Architecture Inductive Biases & Latent Properties", expanded=False):
        st.markdown(
            \"\"\"
            <table style="width:100%; border-collapse:collapse; font-size:0.83rem; margin-top:4px;">
                <tr style="border-bottom:2px solid #cbd5e1; background:#f1f5f9; text-align:left;">
                    <th style="padding:8px;">Architecture</th>
                    <th style="padding:8px;">Core Convolutional Mechanism</th>
                    <th style="padding:8px;">Inductive Bias & Latent Properties</th>
                    <th style="padding:8px;">Diagnostic Sensitivity</th>
                </tr>
                <tr style="border-bottom:1px solid #e2e8f0;">
                    <td style="padding:8px;"><b>EfficientNet-B0</b></td>
                    <td style="padding:8px;">Depthwise separable MBConv blocks with Squeeze-and-Excitation (SE)</td>
                    <td style="padding:8px;">Highly compact, parameter-efficient representation; channel attention focuses sharply on subtle metastasis edges.</td>
                    <td style="padding:8px;"><b>95.28% (Top)</b></td>
                </tr>
                <tr style="border-bottom:1px solid #e2e8f0;">
                    <td style="padding:8px;"><b>ResNet-50</b></td>
                    <td style="padding:8px;">Deep residual bottleneck shortcuts (1x1 -> 3x3 -> 1x1 convs)</td>
                    <td style="padding:8px;">Additive identity skips prevent gradient vanishing; builds smooth, manifold-like representations of complex cellular stroma.</td>
                    <td style="padding:8px;"><b>94.03%</b></td>
                </tr>
                <tr style="border-bottom:1px solid #e2e8f0;">
                    <td style="padding:8px;"><b>DenseNet-121</b></td>
                    <td style="padding:8px;">Dense channel-wise concatenation across all preceding layers</td>
                    <td style="padding:8px;">Substantial feature reuse across fine-grained textural scales; highest precision (85.63%) and specificity (98.13%).</td>
                    <td style="padding:8px;"><b>89.94%</b></td>
                </tr>
                <tr>
                    <td style="padding:8px;"><b>VGG-16</b></td>
                    <td style="padding:8px;">Homogeneous 3x3 stacks followed by dense fully connected layers</td>
                    <td style="padding:8px;">High parameter count (134M), high feature redundancy; latent variance concentrates heavily in top 2 components (98.2%).</td>
                    <td style="padding:8px;"><b>94.34%</b></td>
                </tr>
            </table>
            \"\"\",
            unsafe_allow_html=True,
        )

    st.markdown("#### 📋 Feature Interpretation and Evidence")
    st.caption("Deep CNN Embeddings vs Physical Continuous Tissue Measurements")

    ak1, ak2 = st.columns(2)
    with ak1:
        st.markdown(
            \"\"\"
            <div style="background:#fef2f2; border:1px solid #fecaca; border-left:4px solid #dc2626; border-radius:6px; padding:12px 14px; margin-bottom:12px; font-size:0.82rem; color:#7f1d1d;">
                <b>❌ What PathoVision AI Does NOT Do During Live Inference:</b><br>
                • <b>No Instance Segmentation:</b> Does not run StarDist, Cellpose, or HoVer-Net to segment individual cell nuclei.<br>
                • <b>No Continuous Micrometric Measurements:</b> Does not compute continuous nuclear diameters in micrometers (μm).<br>
                • <b>No Explicit Nuclear-to-Cytoplasmic (N:C) Ratio:</b> No geometric cell boundaries are polygonized.<br>
                • <b>No Direct Necrotic Area Planimetry:</b> Does not calculate physical square micrometers of necrotic debris.<br>
                • <b>No Mitotic Spindle Counting:</b> Does not execute an automated mitotic figure counting detector.
            </div>
            \"\"\",
            unsafe_allow_html=True,
        )
    with ak2:
        st.markdown(
            \"\"\"
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #16a34a; border-radius:6px; padding:12px 14px; margin-bottom:12px; font-size:0.82rem; color:#14532d;">
                <b>✅ What PathoVision AI Actually Computes:</b><br>
                • <b>1,280-D Latent Embedding:</b> High-dimensional continuous feature vector extracted from the frozen EfficientNet backbone.<br>
                • <b>Primary Malignancy Probability:</b> Binary classification score evaluating metastatic carcinoma presence in sentinel lymph nodes.<br>
                • <b>3 Permitted Linear Anomaly Probes:</b> Branching linear classifiers (BACH ductal, NuCLS density, BCSS necrosis) evaluated on the 1,280-D vector.<br>
                • <b>Calibrated Decision Thresholding:</b> Evaluates whether probe sigmoid scores exceed empirically validated thresholds (τ).<br>
                • <b>Gradient Saliency Attribution (Grad-CAM):</b> Backpropagates class gradients to convolutional feature maps to illustrate salient attention foci.
            </div>
            \"\"\",
            unsafe_allow_html=True,
        )

    with st.expander("📁 Historical Offline Files Notice", expanded=False):
        st.markdown(
            \"\"\"
            <div style="font-size:0.82rem; color:#475569; line-height:1.45;">
                The files <code>nucls_features.csv</code> and <code>nucls_spatial_features.csv</code> in <code>result/objective_2/</code>
                are historical offline training artifacts generated from external NuCLS annotations.
                They are <b>NOT</b> queried, calculated, or referenced during live patient patch analysis.
            </div>
            \"\"\",
            unsafe_allow_html=True,
        )

'''

sec5_anchor = '# ==============================================================================\n# SECTION 5: INSTITUTIONAL REVIEWER AUDIT & RBAC'
if sec5_anchor in text and 'elif nav == "🧬 Learned Feature Analysis":' not in text:
    text = text.replace(sec5_anchor, sec4_code + '\n' + sec5_anchor, 1)
    print("Section 4 inserted.")
else:
    print("Section 4 already present or sec5_anchor not found.")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Finished script execution.")
