# CLPSO
The general class of meta-heuristic algorithms consists of a series of trial-and-error processes to resource within the constructed population the optimal solution of optimization problems. The word “meta” describes beyond or higher level. The meta-heuristic algorithm is classified as a population-based (or trajectory-based) technique, containing two subpopulation exploitation and exploration phases. The exploitation ability searches for solutions in a local area by using the information on the good local solution. Meanwhile, the exploration constructs the search space on global positions to produce the global optimum. The good balance between the two explorations and exploitation of sample positions defines the preferable property for any meta-heuristic methods in order to escape the premature convergence of local optima and hence obtain the likelihood of accurate optima. 

Many meta-heuristic algorithms have been introduced with underlying exploitation and exploration abilities. The particle swarm optimization (PSO), being a swarm-intelligence approach, emulates the movement or social behavior of a bird flock. The PSO constructs a set of particles in the population, where their positions are iteratively updated through the movement (velocity functions) learned from the global best particle. However, the premature local optima are often encountered by the standard PSO method as its social update components do not work sufficiently. Various new techniques have been incorporated within the original PSO to enhance its global search ability and overcome the local optimal pitfalls. Our recent work successfully applied an outstanding variant version of the PSO, called comprehensive learning particle swarm optimization (CLPSO) [1] for various engineering application problems [2-5]. In the CLPSO, the learning technique enables the cross positions between the sets of best swarm particles in each dimensional space leading to the likelihood of overcoming locally optimal searches and premature termination of the undesired non-optimal but feasible solutions. The proposed scheme follows suit the learning probability function to define the cooperative responses among swarm populations. 

Reference Paper: 
[1] Liang JJ, Qin AK, Suganthan PN, Baskar S (2006). Comprehensive learning particle swarm optimizer for global optimization of multimodal functions. _IEEE Transactions on Evolutionary Computation_. 10, 281-95.

![1](https://user-images.githubusercontent.com/65479151/226549989-bc7d092e-2156-4e2b-9d51-808cfbd94ece.jpg)

Author's Publications:
[2] Van Thu Huynh, Tangaramvong S, Limkatanyu S, Xuan HN (2022). Two-phase ESO and comprehensive learning PSO method for structural optimization with discrete steel sections. _Advances in Engineering Software_. 167:103102.

[3] Van Thu Huynh, Tangaramvong S, S. Muong, and P. T. Van (2022), Combined Gaussian local search and enhanced comprehensive learning PSO algorithm for size and shape optimization of truss structures, _Buildings_, 12-1976. 

[4] Ei Cho Pyone, Van Thu Huynh, Tangaramvong, S., Linh Van Hong Bui, & Wei Gao (2023). Comprehensive Learning Phasor Particle Swarm Optimization of Structures under Limited Natural Frequency Conditions. _Acta Mechanica Sinica_, 39, 722386.

[5] Van Thu Huynh, Tangaramvong, S., Do, B., Gao, W., & Limkatanyu, S. (2023). Sequential Most Probable Point Update Combining Gaussian Process and Comprehensive Learning PSO for Structural Reliability-Based Design Optimization. _Reliability Engineering & System Safety_, 109164.
