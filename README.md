# Lattice Boltzmann Method Fluid Solver

A 2D lattice boltzmann method (D2Q9, BGK) fluid solver in C++.

This serves as an entertainment of my interests in statistical mechanics, computational physcis, and high-performance c++.

## Curent Status
- CPU Solver
- Drawing Method to represent a test run
- A stable looking test run on cylinder

![Bluff cylinder developing a Von Karman Street](image-1.png)

## Next Steps
- Validate what I am currently getting against:
    - Pouiselle Flow
    - Cylinder Strouhal Numbers
    - Schafer Turek Benchmark
- Port solver to my laptops GPU
- OpenGL representation of GPU runs

As with is the general course for  LBM solvers, I am keeping it within a fairly incompressible regime. Ma < 0.3 (Ma ~ 0.17) for stable flow so that the propogation of our pressure doesnt have to worry about the flow catching up with it. Also, error will scale with Ma^2.

## Validation

My validation takes 3 pieces. 

### Poiseiulle Flow

Given how difficult NS can be, we look for simple analytic solutions to verify the behavior of our solver. Assume we have an long unobstructed pipe of fluid that has reached steady state flow. I highly recommend Brian Storey's video on this. His walkthrough is very clear and concise. He doesn't know me or anything just thought he made a great video :P. 

[Brian Storey's Video](https://www.youtube.com/watch?v=zsp6ZNB62mc)

For a similar setup, we expect to get back a parabolic velocity profile. That is precisely what check_poiseuille.cpp does. It required an extra forcing term to replicate the -dp/dx (i.e. the constant pressure gradient in the direction of the pipe) but it does replicate it pretty faithfully.

For my viewer, I added the option to visually watch the flow converge to what we expect.

![Screenshot of Poiseuille Flow Validation](image.png)