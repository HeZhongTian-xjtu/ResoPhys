# ResoPhys
Official implementation of ResoPhys (IEEE JBHI). ResoPhys is an unsupervised, plug-and-play framework for rPPG measurement from arbitrary-resolution facial videos. By strategically decoupling resolution handling via adaptive feature capture and restoration, ResoPhys enhances the robustness of existing backbones in arbitrary-resolution scenarios.

https://github.com/HeZhongTian-xjtu/ResoPhys/blob/main/overview.png

ResoPhys processes a facial video through two parallel branches at randomly selected resolutions. In each branch, the Arbitrary-Resolution Feature Capture module uses dynamic, scale-aware convolutions to adaptively extract features from the variable-sized input. Subsequently, the Arbitrary-Resolution Feature Upsampling module intelligently restores these features to a standardized format, recovering crucial sub-pixel details. The entire framework is trained via a Multi-Resolution Contrastive Loss, learning a resolution-invariant representation in an unsupervised manner.
