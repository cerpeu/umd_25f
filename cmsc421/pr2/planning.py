"""Planning (Chapters 10-11)"""

import copy
import itertools
from collections import deque, defaultdict
from functools import reduce as _reduce
import numpy as np

import search
from csp import sat_up, NaryCSP, Constraint, ac_search_solver, is_constraint
from logic import FolKB, conjuncts, unify_mm, associate, SAT_plan, cdcl_satisfiable
from search import Node
from utils import Expr, expr, first

class PlanningProblem:
    """
    Planning Domain Definition Language (PlanningProblem) used to define a search problem.
    It stores states in a knowledge base consisting of first order logic statements.
    The conjunction of these logical statements completely defines a state.
    """

    def __init__(self, initial, goals, actions, domain=None):
        self.initial = self.convert(initial) if domain is None else self.convert(initial) + self.convert(domain)
        self.goals = self.convert(goals)
        self.actions = actions
        self.domain = domain

    def convert(self, clauses):
        """Converts strings into exprs"""
        if not isinstance(clauses, Expr):
            if len(clauses) > 0:
                clauses = expr(clauses)
            else:
                clauses = []
        try:
            clauses = conjuncts(clauses)
        except AttributeError:
            pass

        new_clauses = []
        for clause in clauses:
            if clause.op == '~':
                new_clauses.append(expr('Not' + str(clause.args[0])))
            else:
                new_clauses.append(clause)
        return new_clauses

    def expand_fluents(self, name=None):

        kb = None
        if self.domain:
            kb = FolKB(self.convert(self.domain))
            for action in self.actions:
                if action.precond:
                    for fests in set(action.precond).union(action.effect).difference(self.convert(action.domain)):
                        if fests.op[:3] != 'Not':
                            kb.tell(expr(str(action.domain) + ' ==> ' + str(fests)))

        objects = set(arg for clause in set(self.initial + self.goals) for arg in clause.args)
        fluent_list = []
        if name is not None:
            for fluent in self.initial + self.goals:
                if str(fluent) == name:
                    fluent_list.append(fluent)
                    break
        else:
            fluent_list = list(map(lambda fluent: Expr(fluent[0], *fluent[1]),
                                   {fluent.op: fluent.args for fluent in self.initial + self.goals +
                                    [clause for action in self.actions for clause in action.effect if
                                     clause.op[:3] != 'Not']}.items()))

        expansions = []
        for fluent in fluent_list:
            for permutation in itertools.permutations(objects, len(fluent.args)):
                new_fluent = Expr(fluent.op, *permutation)
                if (self.domain and kb.ask(new_fluent) is not False) or not self.domain:
                    expansions.append(new_fluent)

        return expansions


    def expand_actions(self, name=None):
        "Generate all possible actions with variable bindings for precondition selection heuristic"

        has_domains = all(action.domain for action in self.actions if action.precond)
        kb = None
        if has_domains:
            kb = FolKB(self.initial)
            for action in self.actions:
                if action.precond:
                    kb.tell(expr(str(action.domain) + ' ==> ' + str(action)))

        objects = set(arg for clause in self.initial for arg in clause.args)
        expansions = []
        action_list = []
        if name is not None:
            for action in self.actions:
                if str(action.name) == name:
                    action_list.append(action)
                    break
        else:
            action_list = self.actions

        for action in action_list:
            for permutation in itertools.permutations(objects, len(action.args)):
                bindings = unify_mm(Expr(action.name, *action.args), Expr(action.name, *permutation))
                if bindings is not None:
                    new_args = []
                    for arg in action.args:
                        if arg in bindings:
                            new_args.append(bindings[arg])
                        else:
                            new_args.append(arg)
                    new_expr = Expr(str(action.name), *new_args)
                    if (has_domains and kb.ask(new_expr) is not False) or (
                            has_domains and not action.precond) or not has_domains:
                        new_preconds = []
                        for precond in action.precond:
                            new_precond_args = []
                            for arg in precond.args:
                                if arg in bindings:
                                    new_precond_args.append(bindings[arg])
                                else:
                                    new_precond_args.append(arg)
                            new_precond = Expr(str(precond.op), *new_precond_args)
                            new_preconds.append(new_precond)
                        new_effects = []
                        for effect in action.effect:
                            new_effect_args = []
                            for arg in effect.args:
                                if arg in bindings:
                                    new_effect_args.append(bindings[arg])
                                else:
                                    new_effect_args.append(arg)
                            new_effect = Expr(str(effect.op), *new_effect_args)
                            new_effects.append(new_effect)
                        expansions.append(Action(new_expr, new_preconds, new_effects))
        return expansions

    def is_strips(self):
        """
        Returns True if the problem does not contain negative literals in preconditions and goals
        """
        return (all(clause.op[:3] != 'Not' for clause in self.goals) and
                all(clause.op[:3] != 'Not' for action in self.actions for clause in action.precond))

    def goal_test(self):
        """Checks if the goals have been reached"""
        return all(goal in self.initial for goal in self.goals)

    def act(self, action):
        """
        Performs the action given as argument.
        Note that action is an Expr like expr('Remove(Glass, Table)') or expr('Eat(Sandwich)')
        """
        action_name = action.op
        args = action.args
        list_action = first(a for a in self.actions if a.name == action_name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action_name))
        if not list_action.check_precond(self.initial, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        self.initial = list_action(self.initial, args).clauses


class Action:
    """
    Defines an action schema using preconditions and effects.
    Use this to describe actions in PlanningProblem.
    action is an Expr where variables are given as arguments(args).
    Precondition and effect are both lists with positive and negative literals.
    Negative preconditions and effects are defined by adding a 'Not' before the name of the clause
    Example:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("NotEaten(food)")]
    effect = [expr("Eaten(food)"), expr("Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, action, precond, effect, domain=None):
        if isinstance(action, str):
            action = expr(action)
        self.name = action.op
        self.args = action.args
        self.precond = self.convert(precond) if domain is None else self.convert(precond) + self.convert(domain)
        self.effect = self.convert(effect)
        self.domain = domain

    def __call__(self, kb, args):
        return self.act(kb, args)

    def __repr__(self):
        return '{}'.format(Expr(self.name, *self.args))

    def convert(self, clauses):
        """Converts strings into Exprs"""
        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                if clauses[i].op == '~':
                    clauses[i] = expr('Not' + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            clauses = clauses.replace('~', 'Not')
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def relaxed(self):
        """
        Removes delete list from the action by removing all negative literals from action's effect
        """
        return Action(Expr(self.name, *self.args), self.precond,
                      list(filter(lambda effect: effect.op[:3] != 'Not', self.effect)))

    def substitute(self, e, args):
        "Replaces variables in expression with their respective Propositional symbol"

        new_args = list(e.args)
        for num, x in enumerate(e.args):
            for i, _ in enumerate(self.args):
                if self.args[i] == x:
                    new_args[num] = args[i]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        "Checks if the precondition is satisfied in the current state"

        if isinstance(kb, list):
            kb = FolKB(kb)
        for clause in self.precond:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        return True

    def act(self, kb, args):
        """Executes the action on the state's knowledge base"""

        if isinstance(kb, list):
            kb = FolKB(kb)

        if not self.check_precond(kb, args):
            raise Exception('Action pre-conditions not satisfied')
        for clause in self.effect:
            kb.tell(self.substitute(clause, args))
            if clause.op[:3] == 'Not':
                new_clause = Expr(clause.op[3:], *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))
            else:
                new_clause = Expr('Not' + clause.op, *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))

        return kb


def goal_test(goals, state):
    """Generic goal testing helper function"""

    if isinstance(state, list):
        kb = FolKB(state)
    else:
        kb = state
    return all(kb.ask(q) is not False for q in goals)


class ForwardPlan(search.Problem):
    """
    [Section 10.2.1]
    Forward state-space search
    """

    def __init__(self, planning_problem):
        super().__init__(associate('&', planning_problem.initial), associate('&', planning_problem.goals))
        self.planning_problem = planning_problem
        self.expanded_actions = self.planning_problem.expand_actions()

    def actions(self, state):
        return [action for action in self.expanded_actions if all(pre in conjuncts(state) for pre in action.precond)]

    def result(self, state, action):
        return associate('&', action(conjuncts(state), action.args).clauses)

    def goal_test(self, state):
        return all(goal in conjuncts(state) for goal in self.planning_problem.goals)

    def h(self, state):
        """
        Computes ignore delete lists heuristic by creating a relaxed version of the original problem (we can do that
        by removing the delete lists from all actions, i.e. removing all negative literals from effects) that will be
        easier to solve through GraphPlan and where the length of the solution will serve as a good heuristic.
        """
        relaxed_planning_problem = PlanningProblem(initial=state.state,
                                                   goals=self.goal,
                                                   actions=[action.relaxed() for action in
                                                            self.planning_problem.actions])
        try:
            # Relies upon GraphPlan exiting when leveled off
            return len(linearize(GraphPlan(relaxed_planning_problem).execute()))
        except:
            return np.inf


class BackwardPlan(search.Problem):
    """
    [Section 10.2.2]
    Backward relevant-states search
    """

    def __init__(self, planning_problem):
        super().__init__(associate('&', planning_problem.goals), associate('&', planning_problem.initial))
        self.planning_problem = planning_problem
        self.expanded_actions = self.planning_problem.expand_actions()

    def actions(self, subgoal):
        """
        Returns True if the action is relevant to the subgoal, i.e.:
        - the action achieves an element of the effects
        - the action doesn't delete something that needs to be achieved
        - the preconditions are consistent with other subgoals that need to be achieved
        """

        def negate_clause(clause):
            return Expr(clause.op.replace('Not', ''), *clause.args) if clause.op[:3] == 'Not' else Expr(
                'Not' + clause.op, *clause.args)

        subgoal = conjuncts(subgoal)
        return [action for action in self.expanded_actions if
                (any(prop in action.effect for prop in subgoal) and
                 not any(negate_clause(prop) in subgoal for prop in action.effect) and
                 not any(negate_clause(prop) in subgoal and negate_clause(prop) not in action.effect
                         for prop in action.precond))]

    def result(self, subgoal, action):
        # g' = (g - effects(a)) + preconds(a)
        return associate('&', set(set(conjuncts(subgoal)).difference(action.effect)).union(action.precond))

    def goal_test(self, subgoal):
        return all(goal in conjuncts(self.goal) for goal in conjuncts(subgoal))

    def h(self, subgoal):
        """
        Computes ignore delete lists heuristic by creating a relaxed version of the original problem (we can do that
        by removing the delete lists from all actions, i.e. removing all negative literals from effects) that will be
        easier to solve through GraphPlan and where the length of the solution will serve as a good heuristic.
        """
        relaxed_planning_problem = PlanningProblem(initial=self.goal,
                                                   goals=subgoal.state,
                                                   actions=[action.relaxed() for action in
                                                            self.planning_problem.actions])
        try:
            return len(linearize(GraphPlan(relaxed_planning_problem).execute()))
        except:
            return np.inf


def CSPlan(planning_problem, solution_length, CSP_solver=ac_search_solver, arc_heuristic=sat_up):
    """
    [Section 10.4.3]
    Planning as Constraint Satisfaction Problem
    """

    def st(var, stage):
        """Returns a string for the var-stage pair that can be used as a variable"""
        return str(var) + "_" + str(stage)

    def if_(v1, v2):
        """If the second argument is v2, the first argument must be v1"""

        def if_fun(x1, x2):
            return x1 == v1 if x2 == v2 else True

        if_fun.__name__ = "if the second argument is " + str(v2) + " then the first argument is " + str(v1) + " "
        return if_fun

    def eq_if_not_in_(actset):
        """First and third arguments are equal if action is not in actset"""

        def eq_if_not_in(x1, a, x2):
            return x1 == x2 if a not in actset else True

        eq_if_not_in.__name__ = "first and third arguments are equal if action is not in " + str(actset) + " "
        return eq_if_not_in

    expanded_actions = planning_problem.expand_actions()
    fluent_values = planning_problem.expand_fluents()
    for horizon in range(solution_length):
        act_vars = [st('action', stage) for stage in range(horizon + 1)]
        domains = {av: list(map(lambda action: expr(str(action)), expanded_actions)) for av in act_vars}
        domains.update({st(var, stage): {True, False} for var in fluent_values for stage in range(horizon + 2)})
        # initial state constraints
        constraints = [Constraint((st(var, 0),), is_constraint(val))
                       for (var, val) in {expr(str(fluent).replace('Not', '')):
                                              True if fluent.op[:3] != 'Not' else False
                                          for fluent in planning_problem.initial}.items()]
        constraints += [Constraint((st(var, 0),), is_constraint(False))
                        for var in {expr(str(fluent).replace('Not', ''))
                                    for fluent in fluent_values if fluent not in planning_problem.initial}]
        # goal state constraints
        constraints += [Constraint((st(var, horizon + 1),), is_constraint(val))
                        for (var, val) in {expr(str(fluent).replace('Not', '')):
                                               True if fluent.op[:3] != 'Not' else False
                                           for fluent in planning_problem.goals}.items()]
        # precondition constraints
        constraints += [Constraint((st(var, stage), st('action', stage)), if_(val, act))
                        # st(var, stage) == val if st('action', stage) == act
                        for act, strps in {expr(str(action)): action for action in expanded_actions}.items()
                        for var, val in {expr(str(fluent).replace('Not', '')):
                                             True if fluent.op[:3] != 'Not' else False
                                         for fluent in strps.precond}.items()
                        for stage in range(horizon + 1)]
        # effect constraints
        constraints += [Constraint((st(var, stage + 1), st('action', stage)), if_(val, act))
                        # st(var, stage + 1) == val if st('action', stage) == act
                        for act, strps in {expr(str(action)): action for action in expanded_actions}.items()
                        for var, val in {expr(str(fluent).replace('Not', '')): True if fluent.op[:3] != 'Not' else False
                                         for fluent in strps.effect}.items()
                        for stage in range(horizon + 1)]
        # frame constraints
        constraints += [Constraint((st(var, stage), st('action', stage), st(var, stage + 1)),
                                   eq_if_not_in_(set(map(lambda action: expr(str(action)),
                                                         {act for act in expanded_actions if var in act.effect
                                                          or Expr('Not' + var.op, *var.args) in act.effect}))))
                        for var in fluent_values for stage in range(horizon + 1)]
        csp = NaryCSP(domains, constraints)
        sol = CSP_solver(csp, arc_heuristic=arc_heuristic)
        if sol:
            return [sol[a] for a in act_vars]


def SATPlan(planning_problem, solution_length, SAT_solver=cdcl_satisfiable):
    """
    [Section 10.4.1]
    Planning as Boolean satisfiability
    """

    def expand_transitions(state, actions):
        state = sorted(conjuncts(state))
        for action in filter(lambda act: act.check_precond(state, act.args), actions):
            transition[associate('&', state)].update(
                {Expr(action.name, *action.args):
                     associate('&', sorted(set(filter(lambda clause: clause.op[:3] != 'Not',
                                                      action(state, action.args).clauses))))
                     if planning_problem.is_strips()
                     else associate('&', sorted(set(action(state, action.args).clauses)))})
        for state in transition[associate('&', state)].values():
            if state not in transition:
                expand_transitions(expr(state), actions)

    transition = defaultdict(dict)
    expand_transitions(associate('&', planning_problem.initial), planning_problem.expand_actions())

    return SAT_plan(associate('&', sorted(planning_problem.initial)), transition,
                    associate('&', sorted(planning_problem.goals)), solution_length, SAT_solver=SAT_solver)


class Level:
    """
    Contains the state of the planning problem
    and exhaustive list of actions which use the
    states as pre-condition.
    """

    def __init__(self, kb):
        """Initializes variables to hold state and action details of a level"""

        self.kb = kb
        # current state
        self.current_state = kb.clauses
        
        # current action to state link
        # Action -> preconditions for that action
        self.current_action_links = {}
        
        # current state to action link
        # Precondition -> what is applicable (actions)
        self.current_state_links = {}
        
        # current action to next state link (E.g. Go(Home, HW) --> At(HW) and NotAt(Home))
        # aka forward link in time (dependency)
        self.next_action_links = {}

        # next state to current action link (E.g. NotAt(Home): [Go(Home, HW), Go(Home, SM)])
        # aka backwards link in time (dependency)
        self.next_state_links = {}
        
        # mutually exclusive actions
        self.action_mutexes = []

        # mutually exclusive states
        self.state_mutexes = []
        
    def __call__(self, actions, objects):
        self.build(actions, objects)
        self.find_mutex()
        
    def __str__(self):
        state_str = ", ".join(str(s) for s in self.current_state)
        action_str = ", ".join(str(a) for a in self.current_action_links.keys())
        mutex_str = ", ".join([str(m) for m in self.action_mutexes])
        return (
            f"<Level>\n"
            f"  Current State: {{{state_str}}}\n"
            f"  Actions: {{{action_str}}}\n"
            f"  Mutex: {{{mutex_str}}}\n"
        )

    __repr__ = __str__

    def separate(self, e):
        """Separates an iterable of elements into positive and negative parts"""

        positive = []
        negative = []
        for clause in e:
            if clause.op[:3] == 'Not':
                negative.append(clause)
            else:
                positive.append(clause)
        return positive, negative

    
    def find_mutex(self):
        "Finds mutually exclusive actions"

        self.action_mutexes = [] # clear out effects from state mutex prior computation

        # Competing Needs - two actions are mutex if any of their preconditions are mutex at the previous state level
        for a1, a2 in itertools.combinations(self.current_action_links.keys(), 2):
            preconds_a1 = self.current_action_links[a1]
            preconds_a2 = self.current_action_links[a2]

            if any({p, q} in self.state_mutexes for p in preconds_a1 for q in preconds_a2):
                mutex_pair = {a1, a2}
                if mutex_pair not in self.action_mutexes:
                    self.action_mutexes.append(mutex_pair)

        # Interference AND Inconsistent Effects Mutex Calculation
        for a1, a2 in itertools.combinations(self.next_action_links.keys(), 2):
            preconds_a1 = self.current_action_links.get(a1, [])
            preconds_a2 = self.current_action_links.get(a2, [])
            effects_a1 = self.next_action_links.get(a1, [])
            effects_a2 = self.next_action_links.get(a2, [])

            interference = False
            # Interference Check
            for p1 in preconds_a1:
                if p1.predicate_negate() in effects_a2:
                    interference = True
            for p2 in preconds_a2:
                if p2.predicate_negate() in effects_a1:
                    interference = True

            # Inconsistent Effects Check
            for e1 in effects_a1:
                if e1.predicate_negate() in effects_a2:
                    interference = True
            for e2 in effects_a2:
                if e2.predicate_negate() in effects_a1:
                    interference = True

            if interference:
                mutex_pair = {a1, a2}
                if mutex_pair not in self.action_mutexes:
                    self.action_mutexes.append(mutex_pair)

    def populate_prop_mutexes(self):
        "Compute the next level's proposition mutexes based on our current action mutexes"

        # Inconsistent support - two props cannot be true given competing supporting actions
        state_mutex = []
        next_state_pairs = itertools.combinations(self.next_state_links.keys(), 2)
        for next_state_pair in list(next_state_pairs):
            s1, s2 = list(next_state_pair)
            acts_to_s1 = self.next_state_links.get(s1, [])
            acts_to_s2 = self.next_state_links.get(s2, [])

            # ensure our mutexes only apply to pairs, not single states. 
            if acts_to_s1 == [] or acts_to_s2 == []:
                continue

            # if any two actions that lead to these states is not mutex, do not add a mutex to these states.
            if all([{a1,a2} in self.action_mutexes or {a2,a1} in self.action_mutexes for a1 in acts_to_s1 for a2 in acts_to_s2]):
                mutex_pair = {s1, s2}
                if mutex_pair not in state_mutex:
                    state_mutex.append(mutex_pair)

        # If there are pairs of propositions that are negations of each other, they need to be mutex
        for s1i in range(len(self.current_state)):
            for s2i in range(s1i,len(self.current_state)):
                s1, s2 = self.current_state[s1i], self.current_state[s2i]
                if repr(s2)[0:3] == "Not" and repr(s1) == repr(s2)[3:] or repr(s1)[0:3] == "Not" and repr(s1)[3:] == repr(s2):
                    mutex_pair = {s1, s2}
                    if mutex_pair not in state_mutex:
                        state_mutex.append(mutex_pair)

        return state_mutex

    def prune_invalid_actions(self):
        """Remove actions whose own preconditions are mutex (unsupportable)."""
        to_remove = []

        # Normalize state mutex set for fast membership checks:
        state_mutex_lookup = set()
        for m in self.state_mutexes:
            state_mutex_lookup.add(frozenset(m))

        for action, preconds in list(self.current_action_links.items()):
            invalid = False

            for p1, p2 in itertools.combinations(preconds, 2):
                if frozenset({p1, p2}) in state_mutex_lookup:
                    invalid = True
                    break

            if invalid:
                to_remove.append(action)

        # Remove invalid actions from all mappings
        for action in to_remove:
            # forward mappings
            self.current_action_links.pop(action, None)
            self.next_action_links.pop(action, None)

            # reverse mapping: state -> actions (current_state_links)
            for precond in list(self.current_state_links.keys()):
                actions_for_pre = self.current_state_links.get(precond, [])
                if action in actions_for_pre:
                    actions_for_pre.remove(action)
                    # if no more actions support this precond, drop the key
                    if not actions_for_pre:
                        self.current_state_links.pop(precond, None)
                    else:
                        self.current_state_links[precond] = actions_for_pre

            # reverse mapping: next_state -> actions (next_state_links)
            for effect in list(self.next_state_links.keys()):
                actions_for_eff = self.next_state_links.get(effect, [])
                if action in actions_for_eff:
                    actions_for_eff.remove(action)
                    if not actions_for_eff:
                        self.next_state_links.pop(effect, None)
                    else:
                        self.next_state_links[effect] = actions_for_eff

    def build(self, actions, objects):
        """Populates the lists and dictionaries containing the state action dependencies"""

        for clause in self.current_state:
            p_expr = Expr('P' + clause.op, *clause.args)
            self.current_action_links[p_expr] = [clause]
            self.next_action_links[p_expr] = [clause]
            self.current_state_links[clause] = [p_expr]
            self.next_state_links[clause] = [p_expr]

        for a in actions:
            num_args = len(a.args)
            possible_args = tuple(itertools.permutations(objects, num_args))

            for arg in possible_args:
                if a.check_precond(self.kb, arg):
                    for num, symbol in enumerate(a.args):
                        if not symbol.op.islower(): # If not lowercase, we care about exact match to object?
                            arg = list(arg)
                            arg[num] = symbol
                            arg = tuple(arg)

                    new_action = a.substitute(Expr(a.name, *a.args), arg)
                    self.current_action_links[new_action] = []

                    for clause in a.precond:
                        new_clause = a.substitute(clause, arg)
                        self.current_action_links[new_action].append(new_clause)
                        if new_clause in self.current_state_links:
                            self.current_state_links[new_clause].append(new_action)
                        else:
                            self.current_state_links[new_clause] = [new_action]

                    self.next_action_links[new_action] = []
                    for clause in a.effect:
                        new_clause = a.substitute(clause, arg)

                        self.next_action_links[new_action].append(new_clause)
                        if new_clause in self.next_state_links:
                            self.next_state_links[new_clause].append(new_action)
                        else:
                            self.next_state_links[new_clause] = [new_action]

    def perform_actions(self):
        "Performs the necessary actions and returns a new Level"

        new_kb = FolKB(list(set(self.next_state_links.keys())))
        return Level(new_kb)


class Graph:
    """
    Contains levels of state and actions
    Used in graph planning algorithm to extract a solution
    """

    def __init__(self, planning_problem):
        self.planning_problem = planning_problem
        self.kb = FolKB(planning_problem.initial)
        self.levels = [Level(self.kb)]
        self.objects = set(arg for clause in self.kb.clauses for arg in clause.args)

    def __call__(self):
        self.expand_graph()

    def __str__(self):
        levels_str = "\n".join(
            f"Level {i}:\n{str(level)}" for i, level in enumerate(self.levels)
        )
        return (
            f"<Graph>\n"
            f"  Objects: {self.objects}\n"
            f"{levels_str}\n"
        )

    __repr__ = __str__

    def expand_graph(self):
        """Expands the graph by a level"""

        last_level = self.levels[-1]
        last_level(self.planning_problem.actions, self.objects) # populate state/actions/mutexes
        last_level.prune_invalid_actions()
        new_level = last_level.perform_actions() # Create new level
        new_level.state_mutexes = last_level.populate_prop_mutexes() # Populate the mutexes for the next state level to come
        self.levels.append(new_level)

    def non_mutex_goals(self, goals, index):
        "Checks whether the goals are mutually exclusive"

        goal_perm = itertools.combinations(goals, 2)
        for g in goal_perm:
            if set(g) in self.levels[index].state_mutexes:
                return False
        return True


class GraphPlan:
    """
    Class for formulation GraphPlan algorithm
    Constructs a graph of state and action space
    Returns solution for the planning problem
    """

    def __init__(self, planning_problem):
        self.graph = Graph(planning_problem)
        self.no_goods = []
        self.solution = []

    def __str__(self):
        sol_str = (
            "No solution found"
            if not self.solution
            else f"Solution with {len(self.solution)} steps"
        )
        return (
            f"<GraphPlan>\n"
            f"  Nogoods: {len(self.no_goods)}\n"
            f"  {sol_str}\n"
        )

    __repr__ = __str__

    def check_leveloff(self):
        """Checks if the graph has leveled off"""
        if len(self.graph.levels) < 2:
            return False

        level = self.graph.levels[-1]
        prev_level = self.graph.levels[-2]

        same_state = set(level.current_state) == set(prev_level.current_state)

        level_mutex = set(frozenset(m) for m in level.state_mutexes)
        prev_mutex = set(frozenset(m) for m in prev_level.state_mutexes)
        same_mutex = level_mutex == prev_mutex
        
        return same_state and same_mutex 

    def _get_preconditions_for(self, action_set, level):
        """Collects all unique preconditions for a given set of actions in a level"""
        all_preconditions = set()
        for action in action_set:
            preconditions = level.current_action_links.get(action, [])
            all_preconditions.update(preconditions)
        return all_preconditions

    def _find_valid_action_sets(self, goals, level):
        """
        Finds sets of actions in the given level that are not mutually exclusive
        and that collectively satisfy all the goals.
        """
        valid_sets = []

        actions_for_goal = {g: level.next_state_links.get(g, []) for g in goals}
        potential_action_groups = [actions_for_goal[g] for g in goals]

        for action_combination in itertools.product(*potential_action_groups):
            action_set = set(action_combination)

            is_mutex = False
            for a1, a2 in itertools.combinations(action_set, 2):
                if {a1, a2} in level.action_mutexes:
                    is_mutex = True
                    break

            if not is_mutex and action_set not in valid_sets:
                valid_sets.append(action_set)

        return valid_sets

    def extract_solution(self, goals):
        """
        Primary method to start the solution extraction process.
        It calls the recursive helper and returns the final plan.
        """
        return self._extract_solution_recursive(set(goals), len(self.graph.levels)-1)

    # Place this within your GraphPlan class, replacing the previous version
    def _extract_solution_recursive(self, goals, level_index):
        """
        Recursively searches for a plan backwards from a given level using negative indexing.

        Args:
            goals (set): The set of goal propositions to satisfy.
            level_index (int): level index
        """

        #BASE CASE: We've recursed back to the initial proposition layer (Level 0).
        #The 'goals' at this point are the preconditions for the very first set of actions.
        #We must check if they exist in the initial state. No further recursion is needed.

        if level_index == 0:
            initial_state = set(self.graph.levels[0].current_state)
            if goals.issubset(initial_state):
                return []  # Success! Return the empty plan to be built upon.
            else:
                return None  # Failure. This path is invalid as preconditions are not met.

        # MEMOIZATION: Check if we've already proven this subproblem is unsolvable.
        # TODO: ADD SUPERSET CHECK
        if (level_index, frozenset(goals)) in self.no_goods:
            return None

        # RECURSIVE STEP:
        # To satisfy goals at `level_index`, we need to find a suitable set of non-mutex actions
        # from the *previous* level's action layer.
        action_level = self.graph.levels[level_index - 1]
        valid_action_sets = self._find_valid_action_sets(goals, action_level)

        # Iterate through each valid action set and try to find a path.
        for action_set in valid_action_sets:
            # The new sub-goals are the combined preconditions for this action set.
            new_goals = self._get_preconditions_for(action_set, action_level)

            # Recurse to solve for the new goals at the previous proposition layer.
            sub_plan = self._extract_solution_recursive(new_goals, level_index - 1)

            # If the recursive call succeeded, we have a solution!
            if sub_plan is not None:
                # Append the current level's actions and return the full plan.
                return sub_plan + [list(action_set)]

        # If the loop finishes without a solution, this subproblem is a "no-good".
        nogood_item = (level_index, frozenset(goals))
        if nogood_item not in self.no_goods:
            self.no_goods.append(nogood_item)
        return None

    def goal_test(self, kb):
        goal_achieved = all(kb.ask(q) is not False for q in self.graph.planning_problem.goals)
        return goal_achieved

    def execute(self):
        "Executes the GraphPlan algorithm for the given problem"

        while True:
            self.graph.expand_graph()
            if (self.goal_test(self.graph.levels[-1].kb) and self.graph.non_mutex_goals(
                    self.graph.planning_problem.goals, -1)):

                solution = self.extract_solution(self.graph.planning_problem.goals)
                if solution:
                    return [solution]

            if self.check_leveloff():
                return None

class Linearize:
    """
    Problem wrapper / coordinator that linearizes partially ordered solutions generated by GraphPlan object.
    """

    def __init__(self, planning_problem):
        self.planning_problem = planning_problem

    def filter(self, solution):
        "Filter out persistence actions from a solution"

        new_solution = []
        for section in solution:
            new_section = []
            for operation in section:
                if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                    new_section.append(operation)
            # filter may remove all actions if all actions are persistent
            if new_section != []:
                new_solution.append(new_section)
        return new_solution

    def orderlevel(self, level, planning_problem):
        "Return valid linear order of actions for a given level"

        for permutation in itertools.permutations(level):
            temp = copy.deepcopy(planning_problem)
            count = 0
            for action in permutation:
                try:
                    temp.act(action)
                    count += 1
                except:
                    count = 0
                    temp = copy.deepcopy(planning_problem)
                    continue
            if count == len(permutation):
                return list(permutation), temp
        # identifying a linear ordering for level failed ... return no solution and same planning problem state
        return None, planning_problem

    def execute(self):
        "Finds a total-order solution for a planning graph. Possibly not the only linearization possible."

        graphPlan_solution = GraphPlan(self.planning_problem).execute()

        # Exit if no plan found
        if graphPlan_solution is None:
            return None

        for possible_plan in graphPlan_solution:
            filtered_solution = self.filter(possible_plan)

            ordered_solution = []
            # planning_problem will maintain the current state as we iterate over levels, allowing test application of actions
            planning_problem = self.planning_problem
            for level in filtered_solution:
                level_solution, planning_problem = self.orderlevel(level, planning_problem)
                if not level_solution:
                    # level failed to apply, this plan shouldn't work
                    ordered_solution = None
                    break # technically could try `continue` anyway, but we shouldn't need to

                for element in level_solution:   
                    ordered_solution.append(element)

            if not ordered_solution:
                continue  ## no plan possible from the partial plan at the level
            else:
                break

        return ordered_solution

def linearize(solution):
    """Converts a level-ordered solution into a linear solution"""

    linear_solution = []
    for section in solution[0]:
        for operation in section:
            if not (operation.op[0] == 'P' and operation.op[1].isupper()):
                linear_solution.append(operation)

    return linear_solution


class PartialOrderPlanner:
    """
    [Section 10.13] PARTIAL-ORDER-PLANNER

    Partially ordered plans are created by a search through the space of plans
    rather than a search through the state space. It views planning as a refinement of partially ordered plans.
    A partially ordered plan is defined by a set of actions and a set of constraints of the form A < B,
    which denotes that action A has to be performed before action B.
    To summarize the working of a partial order planner,
    1. An open precondition is selected (a sub-goal that we want to achieve).
    2. An action that fulfils the open precondition is chosen.
    3. Temporal constraints are updated.
    4. Existing causal links are protected. Protection is a method that checks if the causal links conflict
       and if they do, temporal constraints are added to fix the threats.
    5. The set of open preconditions is updated.
    6. Temporal constraints of the selected action and the next action are established.
    7. A new causal link is added between the selected action and the owner of the open precondition.
    8. The set of new causal links is checked for threats and if found, the threat is removed by either promotion or
       demotion. If promotion or demotion is unable to solve the problem, the planning problem cannot be solved with
       the current sequence of actions or it may not be solvable at all.
    9. These steps are repeated until the set of open preconditions is empty.
    """

    def __init__(self, planning_problem):
        self.tries = 1
        self.planning_problem = planning_problem
        self.causal_links = []
        self.start = Action('Start', [], self.planning_problem.initial)
        self.finish = Action('Finish', self.planning_problem.goals, [])
        self.actions = set()
        self.actions.add(self.start)
        self.actions.add(self.finish)
        self.constraints = set()
        self.constraints.add((self.start, self.finish))
        self.agenda = set()
        for precond in self.finish.precond:
            self.agenda.add((precond, self.finish))
        self.expanded_actions = planning_problem.expand_actions()

    def find_open_precondition(self):
        """Find open precondition with the least number of possible actions"""

        number_of_ways = dict()
        actions_for_precondition = dict()
        for element in self.agenda:
            open_precondition = element[0]
            possible_actions = list(self.actions) + self.expanded_actions
            for action in possible_actions:
                for effect in action.effect:
                    if effect == open_precondition:
                        if open_precondition in number_of_ways:
                            number_of_ways[open_precondition] += 1
                            actions_for_precondition[open_precondition].append(action)
                        else:
                            number_of_ways[open_precondition] = 1
                            actions_for_precondition[open_precondition] = [action]

        number = sorted(number_of_ways, key=number_of_ways.__getitem__)

        for k, v in number_of_ways.items():
            if v == 0:
                return None, None, None

        act1 = None
        for element in self.agenda:
            if element[0] == number[0]:
                act1 = element[1]
                break

        if number[0] in self.expanded_actions:
            self.expanded_actions.remove(number[0])

        return number[0], act1, actions_for_precondition[number[0]]

    def find_action_for_precondition(self, oprec):
        """Find action for a given precondition"""

        # either
        #   choose act0 E Actions such that act0 achieves G
        for action in self.actions:
            for effect in action.effect:
                if effect == oprec:
                    return action, 0

        # or
        #   choose act0 E Actions such that act0 achieves G
        for action in self.planning_problem.actions:
            for effect in action.effect:
                if effect.op == oprec.op:
                    bindings = unify_mm(effect, oprec)
                    if bindings is None:
                        break
                    return action, bindings

    def generate_expr(self, clause, bindings):
        """Generate atomic expression from generic expression given variable bindings"""

        new_args = []
        for arg in clause.args:
            if arg in bindings:
                new_args.append(bindings[arg])
            else:
                new_args.append(arg)

        try:
            return Expr(str(clause.name), *new_args)
        except:
            return Expr(str(clause.op), *new_args)

    def generate_action_object(self, action, bindings):
        """Generate action object given a generic action and variable bindings"""

        # if bindings is 0, it means the action already exists in self.actions
        if bindings == 0:
            return action

        # bindings cannot be None
        else:
            new_expr = self.generate_expr(action, bindings)
            new_preconds = []
            for precond in action.precond:
                new_precond = self.generate_expr(precond, bindings)
                new_preconds.append(new_precond)
            new_effects = []
            for effect in action.effect:
                new_effect = self.generate_expr(effect, bindings)
                new_effects.append(new_effect)
            return Action(new_expr, new_preconds, new_effects)

    def cyclic(self, graph):
        """Check cyclicity of a directed graph"""

        new_graph = dict()
        for element in graph:
            if element[0] in new_graph:
                new_graph[element[0]].append(element[1])
            else:
                new_graph[element[0]] = [element[1]]

        path = set()

        def visit(vertex):
            path.add(vertex)
            for neighbor in new_graph.get(vertex, ()):
                if neighbor in path or visit(neighbor):
                    return True
            path.remove(vertex)
            return False

        value = any(visit(v) for v in new_graph)
        return value

    def add_const(self, constraint, constraints):
        """Add the constraint to constraints if the resulting graph is acyclic"""

        if constraint[0] == self.finish or constraint[1] == self.start:
            return constraints

        new_constraints = set(constraints)
        new_constraints.add(constraint)

        if self.cyclic(new_constraints):
            return constraints
        return new_constraints

    def is_a_threat(self, precondition, effect):
        """Check if effect is a threat to precondition"""

        if (str(effect.op) == 'Not' + str(precondition.op)) or ('Not' + str(effect.op) == str(precondition.op)):
            if effect.args == precondition.args:
                return True
        return False

    def protect(self, causal_link, action, constraints):
        """Check and resolve threats by promotion or demotion"""

        threat = False
        for effect in action.effect:
            if self.is_a_threat(causal_link[1], effect):
                threat = True
                break

        if action != causal_link[0] and action != causal_link[2] and threat:
            # try promotion
            new_constraints = set(constraints)
            new_constraints.add((action, causal_link[0]))
            if not self.cyclic(new_constraints):
                constraints = self.add_const((action, causal_link[0]), constraints)
            else:
                # try demotion
                new_constraints = set(constraints)
                new_constraints.add((causal_link[2], action))
                if not self.cyclic(new_constraints):
                    constraints = self.add_const((causal_link[2], action), constraints)
                else:
                    # both promotion and demotion fail
                    print('Unable to resolve a threat caused by', action, 'onto', causal_link)
                    return
        return constraints

    def convert(self, constraints):
        """Convert constraints into a dict of Action to set orderings"""

        graph = dict()
        for constraint in constraints:
            if constraint[0] in graph:
                graph[constraint[0]].add(constraint[1])
            else:
                graph[constraint[0]] = set()
                graph[constraint[0]].add(constraint[1])
        return graph

    def toposort(self, graph):
        """Generate topological ordering of constraints"""

        if len(graph) == 0:
            return

        graph = graph.copy()

        for k, v in graph.items():
            v.discard(k)

        extra_elements_in_dependencies = _reduce(set.union, graph.values()) - set(graph.keys())

        graph.update({element: set() for element in extra_elements_in_dependencies})
        while True:
            ordered = set(element for element, dependency in graph.items() if len(dependency) == 0)
            if not ordered:
                break
            yield ordered
            graph = {element: (dependency - ordered)
                     for element, dependency in graph.items()
                     if element not in ordered}
        if len(graph) != 0:
            raise ValueError('The graph is not acyclic and cannot be linearly ordered')

    def display_plan(self):
        """Display causal links, constraints and the plan"""

        print('Causal Links')
        for causal_link in self.causal_links:
            print(causal_link)

        print('\nConstraints')
        for constraint in self.constraints:
            print(constraint[0], '<', constraint[1])

        print('\nPartial Order Plan')
        print(list(reversed(list(self.toposort(self.convert(self.constraints))))))

    def execute(self, display=True):
        """Execute the algorithm"""

        step = 1
        while len(self.agenda) > 0:
            step += 1
            # select <G, act1> from Agenda
            try:
                G, act1, possible_actions = self.find_open_precondition()
            except IndexError:
                print('Probably Wrong')
                break

            act0 = possible_actions[0]
            # remove <G, act1> from Agenda
            self.agenda.remove((G, act1))

            # For actions with variable number of arguments, use least commitment principle
            # act0_temp, bindings = self.find_action_for_precondition(G)
            # act0 = self.generate_action_object(act0_temp, bindings)

            # Actions = Actions U {act0}
            self.actions.add(act0)

            # Constraints = add_const(start < act0, Constraints)
            self.constraints = self.add_const((self.start, act0), self.constraints)

            # for each CL E CausalLinks do
            #   Constraints = protect(CL, act0, Constraints)
            for causal_link in self.causal_links:
                self.constraints = self.protect(causal_link, act0, self.constraints)

            # Agenda = Agenda U {<P, act0>: P is a precondition of act0}
            for precondition in act0.precond:
                self.agenda.add((precondition, act0))

            # Constraints = add_const(act0 < act1, Constraints)
            self.constraints = self.add_const((act0, act1), self.constraints)

            # CausalLinks U {<act0, G, act1>}
            if (act0, G, act1) not in self.causal_links:
                self.causal_links.append((act0, G, act1))

            # for each A E Actions do
            #   Constraints = protect(<act0, G, act1>, A, Constraints)
            for action in self.actions:
                self.constraints = self.protect((act0, G, act1), action, self.constraints)

            if step > 200:
                print("Couldn't find a solution")
                return None, None

        if display:
            self.display_plan()
        else:
            return self.constraints, self.causal_links


class HLA(Action):
    """
    Define Actions for the real-world (that may be refined further), and satisfy resource
    constraints.
    """
    unique_group = 1

    def __init__(self, action, precond=None, effect=None, duration=0, consume=None, use=None):
        """
        As opposed to actions, to define HLA, we have added constraints.
        duration holds the amount of time required to execute the task
        consumes holds a dictionary representing the resources the task consumes
        uses holds a dictionary representing the resources the task uses
        """
        precond = precond or [None]
        effect = effect or [None]
        super().__init__(action, precond, effect)
        self.duration = duration
        self.consumes = consume or {}
        self.uses = use or {}
        self.completed = False
        # self.priority = -1 #  must be assigned in relation to other HLAs
        # self.job_group = -1 #  must be assigned in relation to other HLAs

    def do_action(self, job_order, available_resources, kb, args):
        """
        An HLA based version of act - along with knowledge base updation, it handles
        resource checks, and ensures the actions are executed in the correct order.
        """
        if not self.has_usable_resource(available_resources):
            raise Exception('Not enough usable resources to execute {}'.format(self.name))
        if not self.has_consumable_resource(available_resources):
            raise Exception('Not enough consumable resources to execute {}'.format(self.name))
        if not self.inorder(job_order):
            raise Exception("Can't execute {} - execute prerequisite actions first".
                            format(self.name))
        kb = super().act(kb, args)  # update knowledge base
        for resource in self.consumes:  # remove consumed resources
            available_resources[resource] -= self.consumes[resource]
        self.completed = True  # set the task status to complete
        return kb

    def has_consumable_resource(self, available_resources):
        """
        Ensure there are enough consumable resources for this action to execute.
        """
        for resource in self.consumes:
            if available_resources.get(resource) is None:
                return False
            if available_resources[resource] < self.consumes[resource]:
                return False
        return True

    def has_usable_resource(self, available_resources):
        """
        Ensure there are enough usable resources for this action to execute.
        """
        for resource in self.uses:
            if available_resources.get(resource) is None:
                return False
            if available_resources[resource] < self.uses[resource]:
                return False
        return True

    def inorder(self, job_order):
        """
        Ensure that all the jobs that had to be executed before the current one have been
        successfully executed.
        """
        for jobs in job_order:
            if self in jobs:
                for job in jobs:
                    if job is self:
                        return True
                    if not job.completed:
                        return False
        return True


class RealWorldPlanningProblem(PlanningProblem):
    """
    Define real-world problems by aggregating resources as numerical quantities instead of
    named entities.

    This class is identical to PDDL, except that it overloads the act function to handle
    resource and ordering conditions imposed by HLA as opposed to Action.
    """

    def __init__(self, initial, goals, actions, jobs=None, resources=None):
        super().__init__(initial, goals, actions)
        self.jobs = jobs
        self.resources = resources or {}

    def act(self, action):
        """
        Performs the HLA given as argument.

        Note that this is different from the superclass action - where the parameter was an
        Expression. For real world problems, an Expr object isn't enough to capture all the
        detail required for executing the action - resources, preconditions, etc need to be
        checked for too.
        """
        args = action.args
        list_action = first(a for a in self.actions if a.name == action.name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action.name))
        self.initial = list_action.do_action(self.jobs, self.resources, self.initial, args).clauses

    def refinements(self, library):  # refinements may be (multiple) HLA themselves ...
        """
        State is a Problem, containing the current state kb library is a
        dictionary containing details for every possible refinement. e.g.:
        {
        'HLA': [
            'Go(Home, SFO)',
            'Go(Home, SFO)',
            'Drive(Home, SFOLongTermParking)',
            'Shuttle(SFOLongTermParking, SFO)',
            'Taxi(Home, SFO)'
            ],
        'steps': [
            ['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'],
            ['Taxi(Home, SFO)'],
            [],
            [],
            []
            ],
        # empty refinements indicate a primitive action
        'precond': [
            ['At(Home) & Have(Car)'],
            ['At(Home)'],
            ['At(Home) & Have(Car)'],
            ['At(SFOLongTermParking)'],
            ['At(Home)']
            ],
        'effect': [
            ['At(SFO) & ~At(Home)'],
            ['At(SFO) & ~At(Home)'],
            ['At(SFOLongTermParking) & ~At(Home)'],
            ['At(SFO) & ~At(SFOLongTermParking)'],
            ['At(SFO) & ~At(Home)']
            ]}
        """
        indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == self.name]
        for i in indices:
            actions = []
            for j in range(len(library['steps'][i])):
                # find the index of the step [j]  of the HLA
                index_step = [k for k, x in enumerate(library['HLA']) if x == library['steps'][i][j]][0]
                precond = library['precond'][index_step][0]  # preconditions of step [j]
                effect = library['effect'][index_step][0]  # effect of step [j]
                actions.append(HLA(library['steps'][i][j], precond, effect))
            yield actions

    def hierarchical_search(self, hierarchy):
        """
        [Figure 11.5]
        'Hierarchical Search, a Breadth First Search implementation of Hierarchical
        Forward Planning Search'
        The problem is a real-world problem defined by the problem class, and the hierarchy is
        a dictionary of HLA - refinements (see refinements generator for details)
        """
        act = Node(self.initial, None, [self.actions[0]])
        frontier = deque()
        frontier.append(act)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()
            # finds the first non primitive hla in plan actions
            (hla, index) = RealWorldPlanningProblem.find_hla(plan, hierarchy)
            prefix = plan.action[:index]
            outcome = RealWorldPlanningProblem(
                RealWorldPlanningProblem.result(self.initial, prefix), self.goals, self.actions)
            suffix = plan.action[index + 1:]
            if not hla:  # hla is None and plan is primitive
                if outcome.goal_test():
                    return plan.action
            else:
                for sequence in RealWorldPlanningProblem.refinements(hla, hierarchy):  # find refinements
                    frontier.append(Node(outcome.initial, plan, prefix + sequence + suffix))

    def result(state, actions):
        """The outcome of applying an action to the current problem"""
        for a in actions:
            if a.check_precond(state, a.args):
                state = a(state, a.args).clauses
        return state

    def angelic_search(self, hierarchy, initial_plan):
        """
        [Figure 11.8]
        A hierarchical planning algorithm that uses angelic semantics to identify and
        commit to high-level plans that work while avoiding high-level plans that don’t.
        The predicate MAKING-PROGRESS checks to make sure that we aren’t stuck in an infinite regression
        of refinements.
        At top level, call ANGELIC-SEARCH with [Act] as the initialPlan.

        InitialPlan contains a sequence of HLA's with angelic semantics

        The possible effects of an angelic HLA in initialPlan are:
        ~ : effect remove
        $+: effect possibly add
        $-: effect possibly remove
        $$: possibly add or remove
        """
        frontier = deque(initial_plan)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()  # sequence of HLA/Angelic HLA's
            opt_reachable_set = RealWorldPlanningProblem.reach_opt(self.initial, plan)
            pes_reachable_set = RealWorldPlanningProblem.reach_pes(self.initial, plan)
            if self.intersects_goal(opt_reachable_set):
                if RealWorldPlanningProblem.is_primitive(plan, hierarchy):
                    return [x for x in plan.action]
                guaranteed = self.intersects_goal(pes_reachable_set)
                if guaranteed and RealWorldPlanningProblem.making_progress(plan, initial_plan):
                    final_state = guaranteed[0]  # any element of guaranteed
                    return RealWorldPlanningProblem.decompose(hierarchy, final_state, pes_reachable_set)
                # there should be at least one HLA/AngelicHLA, otherwise plan would be primitive
                hla, index = RealWorldPlanningProblem.find_hla(plan, hierarchy)
                prefix = plan.action[:index]
                suffix = plan.action[index + 1:]
                outcome = RealWorldPlanningProblem(
                    RealWorldPlanningProblem.result(self.initial, prefix), self.goals, self.actions)
                for sequence in RealWorldPlanningProblem.refinements(hla, hierarchy):  # find refinements
                    frontier.append(
                        AngelicNode(outcome.initial, plan, prefix + sequence + suffix, prefix + sequence + suffix))

    def intersects_goal(self, reachable_set):
        """
        Find the intersection of the reachable states and the goal
        """
        return [y for x in list(reachable_set.keys())
                for y in reachable_set[x]
                if all(goal in y for goal in self.goals)]

    def is_primitive(plan, library):
        """
        checks if the hla is primitive action
        """
        for hla in plan.action:
            indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == hla.name]
            for i in indices:
                if library["steps"][i]:
                    return False
        return True

    def reach_opt(init, plan):
        """
        Finds the optimistic reachable set of the sequence of actions in plan
        """
        reachable_set = {0: [init]}
        optimistic_description = plan.action  # list of angelic actions with optimistic description
        return RealWorldPlanningProblem.find_reachable_set(reachable_set, optimistic_description)

    def reach_pes(init, plan):
        """
        Finds the pessimistic reachable set of the sequence of actions in plan
        """
        reachable_set = {0: [init]}
        pessimistic_description = plan.action_pes  # list of angelic actions with pessimistic description
        return RealWorldPlanningProblem.find_reachable_set(reachable_set, pessimistic_description)

    def find_reachable_set(reachable_set, action_description):
        """
        Finds the reachable states of the action_description when applied in each state of reachable set.
        """
        for i in range(len(action_description)):
            reachable_set[i + 1] = []
            if type(action_description[i]) is AngelicHLA:
                possible_actions = action_description[i].angelic_action()
            else:
                possible_actions = action_description
            for action in possible_actions:
                for state in reachable_set[i]:
                    if action.check_precond(state, action.args):
                        if action.effect[0]:
                            new_state = action(state, action.args).clauses
                            reachable_set[i + 1].append(new_state)
                        else:
                            reachable_set[i + 1].append(state)
        return reachable_set

    def find_hla(plan, hierarchy):
        """
        Finds the the first HLA action in plan.action, which is not primitive
        and its corresponding index in plan.action
        """
        hla = None
        index = len(plan.action)
        for i in range(len(plan.action)):  # find the first HLA in plan, that is not primitive
            if not RealWorldPlanningProblem.is_primitive(Node(plan.state, plan.parent, [plan.action[i]]), hierarchy):
                hla = plan.action[i]
                index = i
                break
        return hla, index

    def making_progress(plan, initial_plan):
        """
        Prevents from infinite regression of refinements

        (infinite regression of refinements happens when the algorithm finds a plan that
        its pessimistic reachable set intersects the goal inside a call to decompose on
        the same plan, in the same circumstances)
        """
        for i in range(len(initial_plan)):
            if plan == initial_plan[i]:
                return False
        return True

    def decompose(hierarchy, plan, s_f, reachable_set):
        solution = []
        i = max(reachable_set.keys())
        while plan.action_pes:
            action = plan.action_pes.pop()
            if i == 0:
                return solution
            s_i = RealWorldPlanningProblem.find_previous_state(s_f, reachable_set, i, action)
            problem = RealWorldPlanningProblem(s_i, s_f, plan.action)
            angelic_call = RealWorldPlanningProblem.angelic_search(problem, hierarchy,
                                                                   [AngelicNode(s_i, Node(None), [action], [action])])
            if angelic_call:
                for x in angelic_call:
                    solution.insert(0, x)
            else:
                return None
            s_f = s_i
            i -= 1
        return solution

    def find_previous_state(s_f, reachable_set, i, action):
        """
        Given a final state s_f and an action finds a state s_i in reachable_set
        such that when action is applied to state s_i returns s_f.
        """
        s_i = reachable_set[i - 1][0]
        for state in reachable_set[i - 1]:
            if s_f in [x for x in RealWorldPlanningProblem.reach_pes(
                    state, AngelicNode(state, None, [action], [action]))[1]]:
                s_i = state
                break
        return s_i


def job_shop_problem():
    """
    [Figure 11.1] JOB-SHOP-PROBLEM

    A job-shop scheduling problem for assembling two cars,
    with resource and ordering constraints.

    Example:
    >>> from planning import *
    >>> p = job_shop_problem()
    >>> p.goal_test()
    False
    >>> p.act(p.jobs[1][0])
    >>> p.act(p.jobs[1][1])
    >>> p.act(p.jobs[1][2])
    >>> p.act(p.jobs[0][0])
    >>> p.act(p.jobs[0][1])
    >>> p.goal_test()
    False
    >>> p.act(p.jobs[0][2])
    >>> p.goal_test()
    True
    >>>
    """
    resources = {'EngineHoists': 1, 'WheelStations': 2, 'Inspectors': 2, 'LugNuts': 500}

    add_engine1 = HLA('AddEngine1', precond='~Has(C1, E1)', effect='Has(C1, E1)', duration=30, use={'EngineHoists': 1})
    add_engine2 = HLA('AddEngine2', precond='~Has(C2, E2)', effect='Has(C2, E2)', duration=60, use={'EngineHoists': 1})
    add_wheels1 = HLA('AddWheels1', precond='~Has(C1, W1)', effect='Has(C1, W1)', duration=30, use={'WheelStations': 1},
                      consume={'LugNuts': 20})
    add_wheels2 = HLA('AddWheels2', precond='~Has(C2, W2)', effect='Has(C2, W2)', duration=15, use={'WheelStations': 1},
                      consume={'LugNuts': 20})
    inspect1 = HLA('Inspect1', precond='~Inspected(C1)', effect='Inspected(C1)', duration=10, use={'Inspectors': 1})
    inspect2 = HLA('Inspect2', precond='~Inspected(C2)', effect='Inspected(C2)', duration=10, use={'Inspectors': 1})

    actions = [add_engine1, add_engine2, add_wheels1, add_wheels2, inspect1, inspect2]

    job_group1 = [add_engine1, add_wheels1, inspect1]
    job_group2 = [add_engine2, add_wheels2, inspect2]

    return RealWorldPlanningProblem(
        initial='Car(C1) & Car(C2) & Wheels(W1) & Wheels(W2) & Engine(E2) & Engine(E2) & ~Has(C1, E1) & ~Has(C2, '
                'E2) & ~Has(C1, W1) & ~Has(C2, W2) & ~Inspected(C1) & ~Inspected(C2)',
        goals='Has(C1, W1) & Has(C1, E1) & Inspected(C1) & Has(C2, W2) & Has(C2, E2) & Inspected(C2)',
        actions=actions,
        jobs=[job_group1, job_group2],
        resources=resources)


def go_to_sfo():
    """Go to SFO Problem"""

    go_home_sfo1 = HLA('Go(Home, SFO)', precond='At(Home) & Have(Car)', effect='At(SFO) & ~At(Home)')
    go_home_sfo2 = HLA('Go(Home, SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')
    drive_home_sfoltp = HLA('Drive(Home, SFOLongTermParking)', precond='At(Home) & Have(Car)',
                            effect='At(SFOLongTermParking) & ~At(Home)')
    shuttle_sfoltp_sfo = HLA('Shuttle(SFOLongTermParking, SFO)', precond='At(SFOLongTermParking)',
                             effect='At(SFO) & ~At(SFOLongTermParking)')
    taxi_home_sfo = HLA('Taxi(Home, SFO)', precond='At(Home)', effect='At(SFO) & ~At(Home)')

    actions = [go_home_sfo1, go_home_sfo2, drive_home_sfoltp, shuttle_sfoltp_sfo, taxi_home_sfo]

    library = {
        'HLA': [
            'Go(Home, SFO)',
            'Go(Home, SFO)',
            'Drive(Home, SFOLongTermParking)',
            'Shuttle(SFOLongTermParking, SFO)',
            'Taxi(Home, SFO)'
        ],
        'steps': [
            ['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'],
            ['Taxi(Home, SFO)'],
            [],
            [],
            []
        ],
        'precond': [
            ['At(Home) & Have(Car)'],
            ['At(Home)'],
            ['At(Home) & Have(Car)'],
            ['At(SFOLongTermParking)'],
            ['At(Home)']
        ],
        'effect': [
            ['At(SFO) & ~At(Home)'],
            ['At(SFO) & ~At(Home)'],
            ['At(SFOLongTermParking) & ~At(Home)'],
            ['At(SFO) & ~At(SFOLongTermParking)'],
            ['At(SFO) & ~At(Home)']]}

    return RealWorldPlanningProblem(initial='At(Home)', goals='At(SFO)', actions=actions), library


class AngelicHLA(HLA):
    """
    Define Actions for the real-world (that may be refined further), under angelic semantics
    """

    def __init__(self, action, precond, effect, duration=0, consume=None, use=None):
        super().__init__(action, precond, effect, duration, consume, use)

    def convert(self, clauses):
        """
        Converts strings into Exprs
        An HLA with angelic semantics can achieve the effects of simple HLA's (add / remove a variable)
        and furthermore can have following effects on the variables:
            Possibly add variable    ( $+ )
            Possibly remove variable ( $- )
            Possibly add or remove a variable ( $$ )

        Overrides HLA.convert function
        """
        lib = {'~': 'Not',
               '$+': 'PosYes',
               '$-': 'PosNot',
               '$$': 'PosYesNot'}

        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                for ch in lib.keys():
                    if clauses[i].op == ch:
                        clauses[i] = expr(lib[ch] + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            for ch in lib.keys():
                clauses = clauses.replace(ch, lib[ch])
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def angelic_action(self):
        """
        Converts a high level action (HLA) with angelic semantics into all of its corresponding high level actions (HLA).
        An HLA with angelic semantics can achieve the effects of simple HLA's (add / remove a variable)
        and furthermore can have following effects for each variable:

            Possibly add variable ( $+: 'PosYes' )        --> corresponds to two HLAs:
                                                                HLA_1: add variable
                                                                HLA_2: leave variable unchanged

            Possibly remove variable ( $-: 'PosNot' )     --> corresponds to two HLAs:
                                                                HLA_1: remove variable
                                                                HLA_2: leave variable unchanged

            Possibly add / remove a variable ( $$: 'PosYesNot' )  --> corresponds to three HLAs:
                                                                        HLA_1: add variable
                                                                        HLA_2: remove variable
                                                                        HLA_3: leave variable unchanged


            example: the angelic action with effects possibly add A and possibly add or remove B corresponds to the
            following 6 effects of HLAs:


            '$+A & $$B':    HLA_1: 'A & B'   (add A and add B)
                            HLA_2: 'A & ~B'  (add A and remove B)
                            HLA_3: 'A'       (add A)
                            HLA_4: 'B'       (add B)
                            HLA_5: '~B'      (remove B)
                            HLA_6: ' '       (no effect)

        """

        effects = [[]]
        for clause in self.effect:
            (n, w) = AngelicHLA.compute_parameters(clause)
            effects = effects * n  # create n copies of effects
            it = range(1)
            if len(effects) != 0:
                # split effects into n sublists (separate n copies created in compute_parameters)
                it = range(len(effects) // n)
            for i in it:
                if effects[i]:
                    if clause.args:
                        effects[i] = expr(str(effects[i]) + '&' + str(
                            Expr(clause.op[w:], clause.args[0])))  # make changes in the ith part of effects
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(
                                str(effects[i + len(effects) // 3]) + '&' + str(Expr(clause.op[6:], clause.args[0])))
                    else:
                        effects[i] = expr(
                            str(effects[i]) + '&' + str(expr(clause.op[w:])))  # make changes in the ith part of effects
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(
                                str(effects[i + len(effects) // 3]) + '&' + str(expr(clause.op[6:])))

                else:
                    if clause.args:
                        effects[i] = Expr(clause.op[w:], clause.args[0])  # make changes in the ith part of effects
                        if n == 3:
                            effects[i + len(effects) // 3] = Expr(clause.op[6:], clause.args[0])

                    else:
                        effects[i] = expr(clause.op[w:])  # make changes in the ith part of effects
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(clause.op[6:])

        return [HLA(Expr(self.name, self.args), self.precond, effects[i]) for i in range(len(effects))]

    def compute_parameters(clause):
        """
        computes n,w

        n = number of HLA effects that the angelic HLA corresponds to
        w = length of representation of angelic HLA effect

                    n = 1, if effect is add
                    n = 1, if effect is remove
                    n = 2, if effect is possibly add
                    n = 2, if effect is possibly remove
                    n = 3, if effect is possibly add or remove

        """
        if clause.op[:9] == 'PosYesNot':
            # possibly add/remove variable: three possible effects for the variable
            n = 3
            w = 9
        elif clause.op[:6] == 'PosYes':  # possibly add variable: two possible effects for the variable
            n = 2
            w = 6
        elif clause.op[:6] == 'PosNot':  # possibly remove variable: two possible effects for the variable
            n = 2
            w = 3  # We want to keep 'Not' from 'PosNot' when adding action
        else:  # variable or ~variable
            n = 1
            w = 0
        return n, w


class AngelicNode(Node):
    """
    Extends the class Node.
    self.action:     contains the optimistic description of an angelic HLA
    self.action_pes: contains the pessimistic description of an angelic HLA
    """

    def __init__(self, state, parent=None, action_opt=None, action_pes=None, path_cost=0):
        super().__init__(state, parent, action_opt, path_cost)
        self.action_pes = action_pes
