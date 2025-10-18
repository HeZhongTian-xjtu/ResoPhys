# ResoPhys
Official implementation of ResoPhys (IEEE JBHI). ResoPhys is an unsupervised, plug-and-play framework for rPPG measurement from arbitrary-resolution facial videos. By strategically decoupling resolution handling via adaptive feature capture and restoration, ResoPhys enhances the robustness of existing backbones in arbitrary-resolution scenarios.

<p align = "center">
<img  src="https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/overview.png" width="900" />
</p>

ResoPhys processes a facial video through two parallel branches at randomly selected resolutions. In each branch, the Arbitrary-Resolution Feature Capture module uses dynamic, scale-aware convolutions to adaptively extract features from the variable-sized input. Subsequently, the Arbitrary-Resolution Feature Upsampling module intelligently restores these features to a standardized format, recovering crucial sub-pixel details. The entire framework is trained via a Multi-Resolution Contrastive Loss, learning a resolution-invariant representation in an unsupervised manner.

## 📊 Experiments and Results

### 1\. Overall Performance

ResoPhys (Ours) achieves state-of-the-art performance, outperforming previous supervised and unsupervised methods across all metrics on the UBFC-rPPG, PURE, and COHFACE datasets.

<p align = "center"> <img src="https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/table_plt/intra_dataset.png" width="800" /> </p>

### 2\. Arbitrary-Resolution Robustness

We tested performance on the COHFACE dataset at various resolutions. While other methods degrade significantly, ResoPhys maintains high accuracy even at an extremely low resolution of 16x16, proving its robustness.

<p align = "center"> <img src="https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/table_plt/arbitrary_resolution.png" width="400" /> </p>

### 3\. Signal Visualization

This visualization shows that our predicted rPPG waveform (red) and its Power Spectral Density (PSD) closely match the ground truth (blue), even as the input video resolution decreases to 32x32 and 16x16.

<p align = "center"> <img src="https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/table_plt/waveform.png" width="400" /> </p>

### 4\. Plug-and-Play Capability

ResoPhys acts as a universal front-end. When plugged into various backbones (CNN-based PhysNet, Transformer-based PhysFormer, Mamba-based RhythmMamba), it consistently and significantly boosts their performance across all resolutions.

<p align = "center"> <img src="https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/table_plt/plug_and_play_testing.png" width="400" /> </p>

### 5\. Ablation Study

This ablation study validates our design. The full model (Exp4) shows the best performance, demonstrating the individual contributions of the ARFU module (vs. Exp1), the Scale-aware CNN (vs. Exp2), and the Mask-guided Map (vs. Exp3).

<p align = "center"> <img src="https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/table_plt/ablation_study.png" width="800" /> </p>

### 6\. Ablation Study: Number of Experts (E)

We analyzed the impact of the number of experts (E) in our dynamic convolutions. The results show that $E=4$ provides the optimal balance of filter diversity and performance, while $E=1, 2$ are insufficient and $E=8$ shows diminishing returns.

<p align = "center"> <img src="https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/table_plt/expert_table.png" width="800" /> </p>

<p align = "center"> <img src="https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/table_plt/expert_plg.png" width="300" /> </p>
