# ACOL-GAN
Clustering is one of the research hotspots of deep learning. Recently, deep generative models provide a new way to achieve clustering. However, from all the proposed models, the performance of the clustering variants based on the Variational Autoencoder(VAE) are superior than that based on the Generative Adversarial Network (GAN), which is mainly because the former allows the data to be multi-mode in latent space but the latter does not do so, making the boundaries of different classes obscure and difficult to distinguish. In this paper, we propose a new GAN-based clustering model named Auto-clustering Output Layer Generative Adversarial Network(ACOL-GAN), which replaces the normal distribution that standard GAN relied on with Gaussian mixture distribution generated by sampling networks and adopts the Auto-clustering Output Layer(ACOL) as the output layer in discriminator. Due to Graph-based Activity Regularization(GAR) terms, softmax nodes of parent-classes are specialized as the competition between each other occurs during training. The experimental results show that ACOL-GAN is superior to most unsupervised clustering algorithms on MNIST, USPS and Fashion-MNIST datasets. 
![image](https://github.com/wusongyuan/ACOL-GAN/rs_image/acc.png
