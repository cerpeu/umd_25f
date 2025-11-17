import sys, os
import collections
from multiprocessing import Process, Queue
from planning import *
from planning_envs import *
from logic import *

import pytest

def test_blocksworld_manual():
    sbw = simple_blocks_world()
    assert sbw.goal_test() == False
    sbw.act(expr('ToTable(A, B)'))
    sbw.act(expr('FromTable(B, A)'))
    assert sbw.goal_test() == False
    sbw.act(expr('FromTable(C, B)'))
    assert sbw.goal_test() == True


def test_logistics_manual():
    init = "In(C1, R1) & In(C2, D1) & In(C3, D2) & In(R1, D1) & Holding(R1)"
    goal_state = "In(C2, D3) & In(C3, D3)"
    P = logisticsPlanCustom(init, goal_state)
    assert P.goal_test() == False
    P.act(expr('PutDown(R1, C1, D1)'))
    P.act(expr('PickUp(R1, C2, D1)'))
    P.act(expr('Move(R1, D1, D3)'))
    P.act(expr('PutDown(R1, C2, D3)'))
    P.act(expr('Move(R1, D3, D2)'))
    P.act(expr('PickUp(R1, C3, D2)'))
    P.act(expr('Move(R1, D2, D3)'))
    assert P.goal_test() == False
    P.act(expr('PutDown(R1, C3, D3)'))
    assert P.goal_test() == True

def test_double_tennis_manual():
    p = double_tennis_problem()

    assert not p.goal_test()
    p.act(expr('Go(A, RightBaseLine, LeftNet)'))
    assert not p.goal_test()
    p.act(expr('Hit(A, Ball, RightBaseLine)'))
    assert not p.goal_test()
    p.act(expr('Go(A, LeftBaseLine, RightBaseLine)'))
    assert not p.goal_test()
    p.act(expr('Go(B, LeftNet, RightNet)'))
    assert p.goal_test()


def test_generalized_blocksworld_manual():
    """
    Manual test for the generalized Blocks World problem constructor.
    This test case involves stacking four blocks (A, B, C, D) into a single tower.
    """
    # 1. Define the problem parameters
    initial_state = 'On(A, Table) & On(B, Table) & On(C, Table) & On(D, Table) & Clear(A) & Clear(B) & Clear(C) & Clear(D)'
    goal_state = 'On(A, B) & On(B, C) & On(C, D)'
    block_names = ['A', 'B', 'C', 'D']
    
    bw_problem = blocks_world(initial_state, goal_state, block_names)
    assert bw_problem.goal_test() == False
    bw_problem.act(expr('Move(C, Table, D)'))
    assert bw_problem.goal_test() == False
    bw_problem.act(expr('Move(B, Table, C)'))
    assert bw_problem.goal_test() == False
    bw_problem.act(expr('Move(A, Table, B)'))
    assert bw_problem.goal_test() == True


def verify_solution(P):
    sol = Linearize(P).execute()
    print(sol)
    assert P.goal_test() == False
    for act in sol:
        P.act(expr(act))
    assert P.goal_test() == True

def test_air_cargo():
    P = air_cargo()
    verify_solution(P)

def test_spare_tire():
    P = spare_tire()
    verify_solution(P)
    
def test_three_block_tower():
    P = three_block_tower()
    verify_solution(P)

def test_simple_blocks_world():
    P = simple_blocks_world()
    verify_solution(P)

def test_shopping_problem():
    P = shopping_problem()
    verify_solution(P)

def test_socks_and_shoes():
    P = socks_and_shoes()
    verify_solution(P)

def test_have_cake_and_eat_cake_too():
    P = have_cake_and_eat_cake_too()
    verify_solution(P)

@pytest.mark.parametrize("goal_state", [
    "In(C1, D1)",
    "In(C1, D2)",
    "In(C1, D1) & In(R1, D2)",
    "In(R1, D2) & In(C1, D1)",
    "In(C1, D1) & In(C3, R1)",
    "In(C1, D1) & In(C3, R1) & In(R1, D3)",
    "In(C1, D1) & In(R1, D3) & In(C3, R1)",
    "In(C1, D1) & In(C3, D3)",
    "In(C1, D1) & In(R1, D2) & In(C3, R1)",
    "In(C1, D1) & In(C3, R1) & In(R1, D3)",
    "In(C1, D1) & In(C2, D3)",
    "In(C3, D1)",
    "In(C2, D3)",
    "In(C2, D3) & In(C3, D3)", \
    "In(C3, D3) & In(C2, D3)", \
    "In(C1, D2) & In(C3, D3)",
    "In(C1, D3) & In(C2, D3) & In(C3, D3)", \
    "In(C1, D2) & In(C3, D3) & In(C2, D1)",
    "In(C3, D3)",
    "In(C1, D2) & In(C3, D3) & In(C2, D3) & In(R1, D1)" \
])
def test_logistics_plan_valid(goal_state):
    """These should yield a valid (non-crashing) plan, even if empty."""
    init = "In(C1, R1) & In(C2, D1) & In(C3, D2) & In(R1, D1) & Holding(R1)"
    P = logisticsPlanCustom(init, goal_state)
    verify_solution(P)

def test_double_tennis_problem_simple():
    P = double_tennis_problem_simple()
    verify_solution(P)
    
    
def test_double_tennis_problem_simple2():
    P = double_tennis_problem_simple2()
    verify_solution(P)
    
def test_double_tennis_problem_simple3():
    P = double_tennis_problem_simple3()
    verify_solution(P)

def test_double_tennis_problem():
    P = double_tennis_problem()
    verify_solution(P)

def test_rush_hour_manual_alt_sequence():
    """
    Provides an alternative manual test for the Rush Hour problem.

    This solution is less efficient but still valid. It interleaves
    the movements of different vehicles and includes an unnecessary move
    to verify that the actions correctly modify the game state without
    breaking the rules.
    """
    # Initialize the problem
    problem = rush_hour()

    # Initial state is not the goal
    assert not problem.goal_test()

    # Step 1: Make an unnecessary move with the BlueCar to show it works.
    # Move BlueCar from (R5, C2) down to (R6, C2).
    # Note: The BlueCar occupies R5 and R6, so we must move it up first to free R6.
    # Let's move it from R5,C2 -> R4,C2 instead.
    problem.act(expr('MoveUpCar(BlueCar, R4, R5, R6, C2)'))
    assert not problem.goal_test(), "Moving the BlueCar should not solve the puzzle."

    # Step 2: Start clearing the main path by moving the GreenTruck.
    # Move GreenTruck down once: R1,C4 -> R2,C4
    problem.act(expr('MoveDownTruck(GreenTruck, R1, R2, R3, R4, C4)'))

    # Step 3: Move the RedCar into the newly available space.
    # Move RedCar right once: R3,C1 -> R3,C2
    problem.act(expr('MoveRightCar(RedCar, R3, C1, C2, C3)'))
    assert not problem.goal_test()

    # Step 4: Continue clearing the path.
    # Move GreenTruck down again: R2,C4 -> R3,C4
    problem.act(expr('MoveDownTruck(GreenTruck, R2, R3, R4, R5, C4)'))
    # Move RedCar right again: R3,C2 -> R3,C3
    problem.act(expr('MoveRightCar(RedCar, R3, C2, C3, C4)'))
    assert not problem.goal_test()

    # Step 5: Final moves to solve the puzzle.
    # Move GreenTruck a final time to completely clear the row: R3,C4 -> R4,C4
    problem.act(expr('MoveDownTruck(GreenTruck, R3, R4, R5, R6, C4)'))
    
    # Move RedCar to the goal position.
    problem.act(expr('MoveRightCar(RedCar, R3, C3, C4, C5)'))
    problem.act(expr('MoveRightCar(RedCar, R3, C4, C5, C6)'))

    # The sequence of actions should now result in the goal state.
    assert problem.goal_test()


# Fails due to massive search space (hangs during building layer 1 maps)
"""
def test_rush_hour():
    P = rush_hour()
    verify_solution(P)
"""

def test_rush_hour_optimized():
    P = rush_hour_optimized()
    verify_solution(P)


def test_planner_leveloff():
    def run_planner_in_queue(problem, queue):
        queue.put(Linearize(problem).execute())

    P = blocks_world(
        'On(A, Table) & On(B, Table) & On(C, Table) & Clear(A) & Clear(B) & Clear(C)',
        'On(A, B) & On(B, C) & On(C, A)',
        ['A', 'B', 'C']
    )
    
    result_queue = Queue()
    proc = Process(target=run_planner_in_queue, args=(P, result_queue))
    proc.start()
    proc.join(timeout=3)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        assert False # Ran for 3 seconds and didn't exit in leveloff
    else:
        result = result_queue.get()
        assert result is None or result == [] or result == [[]]
        
def test_impossible_cake_exits_via_leveloff():
    """
    Verify that GraphPlan terminates and returns None for the impossible cake problem.
    """

    def impossible_cake_problem():
        """
        An impossible planning problem to demonstrate GraphPlan's level-off detection.

        The goal is to both Have(Cake) and Eaten(Cake). However, the only available
        action, Eat(Cake), has the effect of ~Have(Cake). The propositions
        Have(Cake) and Eaten(Cake) will become mutually exclusive at the first
        level, and the graph will quickly level off, proving the goal is unreachable.
        """
        return PlanningProblem(
            initial='Have(Cake) & ~Eaten(Cake)',
            goals='Have(Cake) & Eaten(Cake)',
            actions=[
                Action('Eat(Cake)',
                    precond='Have(Cake)',
                    effect='Eaten(Cake) & ~Have(Cake)')
            ]
        )

    problem = impossible_cake_problem()
    solution = Linearize(problem).execute()
    assert solution is None