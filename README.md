# Intelligent-reflecting-surface-circuit-optimization-using-PSO

# PSO Based IRS Circuit Optimization

## Introduction

This project applies Particle Swarm Optimization (PSO) to optimize the physical circuit parameters of an Intelligent Reflecting Surface (IRS) in a wireless communication system.

Instead of optimizing the phase shift directly, the project optimizes the circuit parameters of each IRS element, including:

* L1
* L2
* C
* R

The circuit parameters determine the reflection coefficient of each IRS element. The resulting reflection coefficients are then used to calculate the achievable data rate of the communication system.

## Main Features

* RLC based IRS reflection model
* Direct optimization of IRS circuit parameters
* Particle Swarm Optimization
* Wireless channel modeling
* Achievable data rate evaluation
* Performance evaluation with different IRS sizes and channel conditions

## System Model

The system consists of:

* Multi antenna Access Point
* Intelligent Reflecting Surface
* Single user receiver
* Direct AP to user channel
* AP to IRS to user reflected channel

Each IRS element is modeled using an equivalent RLC circuit.

The reflection coefficient is calculated from the circuit impedance and is used to evaluate the communication performance.

## Optimization

PSO searches for suitable values of L1, L2, C and R for all IRS elements within predefined physical constraints.

The objective of the optimization is to maximize the achievable data rate.

The detailed mathematical model, algorithm description and experimental analysis are provided in the project report.

## Project Structure

```text
PSO Based IRS Circuit Optimization
|
|-- psocode.py # Main Python script for PSO implementation
|-- Project_PhaseShift_Model.pdf # reference paper
|-- optimization_method_descritption.pdf # Detailed methodology, workflow, and results
|-- README.md
```

## Requirements

Python 3.x

Required packages:

```bash
pip install numpy matplotlib
```

## How to Run

Run the main program:

```bash
python pso.py
```

The program performs PSO based circuit parameter optimization and evaluates the achievable data rate.

## Results

The project evaluates the optimization performance under different IRS sizes, user locations and channel realizations.

Detailed results and analysis are presented in the project report.

## Reference

S. Abeywickrama, R. Zhang, and C. Yuen

Intelligent Reflecting Surface: Practical Phase Shift Model and Beamforming Optimization

## Author

Duc Pham

Hanoi University of Science and Technology

