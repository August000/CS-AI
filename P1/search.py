# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
 
    visited = set()                            #instantiate a set that holds the nodes visited (keeps the algorithms from getting stuck in cycles)
    fringe = util.Stack()                           #instantiate a stack that will be used iterate through the the search problem 
    fringe.push((problem.getStartState(), []))      #adds initial values into the stack that holds tuples (A, [])

    while not fringe.isEmpty():                     #configues a loop that will run untill we have popped all items in the stack. This makes sure that we check each subnode of node A and so on.

        node, actions = fringe.pop()                #we destructure the tuple into it's two values. The node and the actions that led to that node.

        if problem.isGoalState(node):               #checks if the search algorithms has reached a goal state
            return actions

        if node not in visited:                     #checks if the node has been visited before
            visited.add(node)                       #adds the node to the visited set

            for cn, cn_action, cn_cost in problem.getSuccessors(node):      #for the node that we haven't visited we get the sucessors of that node and we iterate through it while breaking the values into cn: child node value, child node actions and child node cost
                if cn not in visited:                                       #checks if we have visited that subnode
                    fringe.push((cn, actions + [cn_action]))                #pushes into the fringe a tuple which includes the node and actions that led to the previous node plus the actions that leads to that subnode

    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
   
    visited = set()                            #instantiate a set that holds the nodes visited (keeps the algorithms from getting stuck in cycles)
    fringe = util.Queue()                           #instantiate a queue that will be used iterate through the the search problem 
    fringe.push((problem.getStartState(), []))      #adds initial values into the queue that holds tuples (A, [])

    while not fringe.isEmpty():                     #configues a loop that will run untill we have popped all items in the queue. This makes sure that we check each subnode in layered order going into the nodes of the superficial layers first.

        node, actions = fringe.pop()                #we destructure the tuple into it's two values. The node and the actions that led to that node.

        if problem.isGoalState(node):               #checks if the search algorithms has reached a goal state
            return actions

        if node not in visited:                     #checks if the node has been visited before
            visited.add(node)                       #adds the node to the visited set

            for cn, cn_action, cn_cost in problem.getSuccessors(node):      #for the node that we haven't visited we get the sucessors of that node and we iterate through it while breaking the values into cn: child node value, child node actions and child node cost
                if cn not in visited:                                       #checks if we have visited that subnode
                    fringe.push((cn, actions + [cn_action]))                #pushes into the fringe a tuple which includes the node and actions that led to the previous node plus the actions that leads to that subnode

    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    start_state = problem.getStartState() # Get start state
    
    # Check if we start at goal state 
    if problem.isGoalState(start_state): 
        return []

    frontier = util.PriorityQueue()  # Priority queue to store frontier states
    frontier.push((start_state, [], 0), 0) # Push start state into the frontier (current state, path taken, cost to this point)
    lowest_cost = {start_state: 0} # Dictionary to record lowest cost to given state

    # loop through frontier until empty
    while not frontier.isEmpty():
        state, actions, cost_so_far = frontier.pop() # pop state with lowest cost

        # check if path is cheapest way to state, skip if not
        if cost_so_far != lowest_cost.get(state, float("inf")):
            continue

        # If goal reached, return path
        if problem.isGoalState(state):
            return actions

        # check successors of current state
        for successor, action, action_cost in problem.getSuccessors(state):
            new_cost = cost_so_far + action_cost

            # If this new path is cheapest to the successor, update lowest_cost and add to frontier
            if new_cost < lowest_cost.get(successor, float("inf")):
                lowest_cost[successor] = new_cost
                frontier.push((successor, actions + [action], new_cost), new_cost)

    # no path to a goal found
    return []




def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic): #similar to ucs but with hueristic
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState() # Get start state
    
    # Check if we start at goal state 
    if problem.isGoalState(start_state): 
        return []

    frontier = util.PriorityQueue()  # Priority queue to store frontier states
    frontier.push((start_state, [], 0), 0 + heuristic(start_state, problem)) # For A* the priority index will be g+h (g=0 for start state)
    lowest_cost = {start_state: 0} # Dictionary to record lowest cost to given state

    # loop through frontier until empty
    while not frontier.isEmpty():
        state, actions, cost_so_far = frontier.pop() # pop state with lowest cost

        # check if path is cheapest way to state, skip if not
        if cost_so_far != lowest_cost.get(state, float("inf")):
            continue

        # If goal reached, return path
        if problem.isGoalState(state):
            return actions

        # check successors of current state
        for successor, action, action_cost in problem.getSuccessors(state):
            new_cost = cost_so_far + action_cost

            # If this new path is cheapest to the successor, update lowest_cost and add to frontier
            if new_cost < lowest_cost.get(successor, float("inf")):
                lowest_cost[successor] = new_cost
                frontier.push((successor, actions + [action], new_cost), new_cost + heuristic(successor, problem))

    # no path to a goal found
    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
