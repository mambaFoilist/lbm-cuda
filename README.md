# Lattice Boltzmann Method Fluid Solver

A 2D lattice boltzmann method (D2Q9, BGK) fluid solver in C++.

This serves as an entertainment of my interests in statistical mechanics, computational physcis, and high-performance c++.

## Curent Status
- CPU Solver
- Drawing Method to represent a test run (It is basic )
- A stable looking test run on cylinder (Need to validate tho)

## Next Steps
- Validate what I am currently getting against:
    - Pouiselle Flow
    - Cylinder Strouhal Numbers
    - Schafer Turek Benchmark
- Port solver to my laptops GPU
- OpenGL representation of GPU runs

As with is the general course for  LBM solvers, I am keeping it within a fairly incompressible regime. Ma < 0.3 (Ma ~ 0.17) for stable flow so that the propogation of our pressure doesnt have to worry about the flow catching up with it. Also, error will scale with Ma^2.