# Project 7: Robots on Ice (Project ID: PG-1)

## Main Focus

- Continuous control
- Policy gradient methods

---

## Scientific Objective

The project investigates the challenges related to the control of continuous-space dynamical systems, in particular
continuous actions.

---

## Problem Description

A robot Zamboni (ice resurfacer) is given a sequence of targets on an ice skating rink. It has to reach each given
target within 20 seconds.

The robot:

- Is holonomic (can move in any direction)
- Has a control frequency of 10 decisions per second
- Operates on an almost frictionless ice surface

Let:

- `m` be the mass of the robot
- `x(t)` be the position (in meters) of the robot in the xy-plane at time `t`
- `v(t)` be the velocity of the robot
- `F(t)` be the force vector applied by the robot actuators

The origin is located at the center of the ice rink.

The system can be approximately modeled using Newton’s laws:

```text
x(t+1) = x(t) + 0.1 * v(t)

v(t+1) = v(t) + 0.1 * F(t) / m
````

The robot:

* Observes its own position and velocity
* Directly controls the applied force
* Can apply forces with magnitude at most 100 Newton

At each time step `t`, the robot receives:

* A penalty `d(t)` equal to the squared distance from the current target
* A penalty:

```text
e(t) = (C / 100) * ||F(t)||^2
```

which is proportional to the squared magnitude of the applied force.

The constant `C`:

* Is positive
* Is smaller than 1
* Models how much energy consumption is penalized compared to the main objective

---

## Tasks

1. Model the problem as an MDP:

    * Define continuous states
    * Define continuous actions
    * Define rewards

2. Model the robot’s control policy using a parametric policy of choice.

3. Implement the agent using a policy gradient method of choice:

    * Find good hyperparameters
    * Train the agent

4. Compare:

    * Final average return
    * Learning curves

for:

* Mass values `m` between 50 and 100 kg
* Cost coefficient `C` between 0 and 1

---

## Challenging Variants

### 1. Ice Rink Boundary

Model the rectangular boundary of the ice rink.

When the robot hits the boundary:

* The episode ends
* The robot should receive a penalty
* Higher-speed collisions should incur larger penalties

Try to size:

* The robot
* The ice rink

in a realistic way.

---

### 2. Friction Model

Model the effect of friction using:

```text
v(t+1) = v(t) + 0.1 * (F(t) / m - b * v(t))
```

where the friction coefficient `b` is between:

```text
10^-2 and 10^-1
```

# Assets

Thank you to the artists for providing the assets under a free use license!

- [Moon Tiles by upklyak](https://www.magnific.com/free-vector/moon-surface-seamless-background-with-craters_24025286.htm)
- [Robot by jcomp](https://www.magnific.com/free-vector/set-ufo-game-vehicle-cartoon-style-illustration_13330355.htm)
- [Stars by juicy_fish](https://www.magnific.com/free-vector/stars_34295176.htm)
