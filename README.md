# PACMANN: Point Adaptive Collocation Method for Artificial Neural Networks

Inspired by classic optimization problems, PACMANN incrementally moves collocation points toward regions of higher residuals using gradient-based optimization algorithms guided by the gradient of the squared residual.

A comprehensive description of PACMANN is provided in the preprint [arXiv:2411.19632v1](https://arxiv.org/abs/2411.19632v1).

Repository DOI: [10.4121/ac3e81b2-987c-42ff-8c5f-2d303b19db17](https://doi.org/10.4121/ac3e81b2-987c-42ff-8c5f-2d303b19db17)


## License
The scripts in this repository are licensed under an **Apache License v2.0** (see [LICENSE](LICENSE))

The scripts and datasets for the [1D Burgers' equation](1D%20Burgers'%20equation), [1D Allen-Cahn equation](1D%20Allen-Cahn%20equation), [2D Navier-Stokes equation](2D%20Navier-Stokes%20equation), and [3D Navier-Stokes equation](3D%20Navier-Stokes%20equation) are based on the examples available in the [DeepXDE repository](https://github.com/lululxvi/deepxde) by Lu et al. (2021). 

Furthermore, the implementations of RAR, RAD, and RAR-D in this repository are derived from the corresponding implementations for the Burgers’ equation available in the [PINN-sampling repository](https://github.com/lu-group/pinn-sampling) by Wu et al. (2023).

Copyright notice:

Technische Universiteit Delft hereby disclaims all copyright interest in the program PACMANN. It is an adaptive collocation sampling method for physics-informed neural networks written by the Author(s).
Henri Werij, Dean of Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2025, C. Visser, A. Heinlein, B. Giovanardi


## Authors

This software has been developed by **Coen Visser** ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0009-0008-2160-1362](https://orcid.org/0009-0008-2160-1362), **Alexander Heinlein** ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0003-1578-8104](https://orcid.org/0000-0003-1578-8104), and **Bianca Giovanardi** ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0000-0001-7768-8542](https://orcid.org/0000-0001-7768-8542), Technische Universiteit Delft


## References
- L. Lu, X. Meng, Z. Mao, G. E. Karniadakis, DeepXDE: A deep learning library for solving differential equations, SIAM Review 63 (1) (2021) 208–228. [doi:10.1137/19M1274067](https://doi.org/10.1137/19M1274067).
- C. Wu, M. Zhu, Q. Tan, Y. Kartha, L. Lu, A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks, Computer Methods in Applied Mechanics and Engineering 403 (2023) 115671. [doi:10.1016/J.CMA.2022.115671](https://doi.org/10.1016/J.CMA.2022.115671).