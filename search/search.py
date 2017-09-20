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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"  
    from util import Stack
    # Here I will create two parallel stacks for 
    # storing the states and actions separately
    # This way I can easily retriever actions with a helper function
    sequence = Stack()
    actions = Stack()
    #adding the start node as a list to the queue for convenient retrieval
    #form of the start node is [(x,y)]
    sequence.push([problem.getStartState()]) 
    #Just adding a None action since no action is needed to get to initial state
    actions.push(["None"])
    
    while not sequence.isEmpty():
        
        cur_route = sequence.pop()
        cur_action = actions.pop()
        # Here I retrieve the current state (x,y) from the state tuple list
        cur_state = cur_route[-1] 
        
        if problem.isGoalState(cur_state):
            print "Solution reached with:", cur_action
            return retrieve_solution(cur_action)
        
        for successor in problem.getSuccessors(cur_state):
            # Path check to eliminate and state that has been visited
            if not successor[0] in cur_route:
                # Generates a new route with successor added 
                new_state = cur_route + [successor[0]] 
                new_action = cur_action + [successor[1]]
                sequence.push(new_state)
                actions.push(new_action)
                
    return False #return False upon no solution found    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    sequence = Queue()
    #adding the start node as a list of tuple
    #(state,action,cost) to the queue for convenient retrieval    
    sequence.push([(problem.getStartState(),'None',0)]) 
    # a dictionary storing seen states and the min cost to that stage
    seen = {problem.getStartState():0}; 
    
    while not sequence.isEmpty():
        
        cur_route = sequence.pop()
        # Here I retrieve the current state (x,y) from the state tuple in form (state,action,cost)
        cur_state = cur_route[-1][0] 
        
        if cost(cur_route) <= seen[cur_state]:
            if problem.isGoalState(cur_state):
                print "solution reached with", cur_route
                return retrieve_direction(cur_route)
            
            for successor in problem.getSuccessors(cur_state):
                new_state = cur_route + [successor]
                # Path check to eliminate and state that has been visited
                if not successor[0] in seen or cost(new_state) < seen[successor[0]]: 
                    sequence.push(new_state)
                    seen[successor[0]] = cost(new_state)
    return False #return False upon no solution found


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    sequence = PriorityQueue()
    #adding the start node as a list of tuple
    #(state,action,cost) to the queue for convenient retrieval    
    sequence.push([(problem.getStartState(),'None',0)],0) 
    # a dictionary storing seen states and the min cost to that stage
    seen = {problem.getStartState():0}; 
    
    while not sequence.isEmpty():
        cur_route = sequence.pop()
        # Here I retrieve the current state (x,y) from the state tuple in form (state,action,cost)
        cur_state = cur_route[-1][0] 
        
        if cost(cur_route) <= seen[cur_state]:
            if problem.isGoalState(cur_state):
                print "solution reached with", cur_route
                return retrieve_direction(cur_route)
            
            for successor in problem.getSuccessors(cur_state):
                new_state = cur_route + [successor]
                # Path check to eliminate and state that has been visited
                if not successor[0] in seen or cost(new_state) < seen[successor[0]]: 
                    new_priority = cost(new_state)
                    sequence.push(new_state,new_priority)
                    seen[successor[0]] = cost(new_state)
         
    return False #return False upon no solution found

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    sequence = PriorityQueue()
    # Computing h(start state)
    init_heuristic = heuristic(problem.getStartState(),problem)
    # F-value = 0 + h(start state), here we are storing the f - value as the priority
    #adding the start node as a list of tuple
    #(state,action,cost) to the queue for convenient retrieval    
    sequence.push([(problem.getStartState(),'None',0)], 0 + init_heuristic) 
    # a dictionary storing seen states and the min-f value to that stage
    seen = {problem.getStartState():0 + init_heuristic}; 
    
    while not sequence.isEmpty():
        cur_route = sequence.pop()
        # Here I retrieve the current state (x,y) from the state tuple in form (state,action,cost)
        cur_state = cur_route[-1][0] 
        # Computing the new heuristic and f-value for the current state
        cur_state_heuristic = heuristic(cur_state,problem)
        cur_state_fvalue = cost(cur_route) + cur_state_heuristic
        
        if cur_state_fvalue <= seen[cur_state]:
            if problem.isGoalState(cur_state):
                print "solution reached with", cur_route
                return retrieve_direction(cur_route)
            
            for successor in problem.getSuccessors(cur_state):
                new_state = cur_route + [successor]
                new_state_heuristic = heuristic(successor[0],problem)
                new_state_fvalue = cost(new_state) + new_state_heuristic
                # Path check to eliminate and state that has been visited
                if not successor[0] in seen or new_state_fvalue < seen[successor[0]]: 
                    new_priority = new_state_fvalue
                    sequence.push(new_state,new_priority)
                    seen[successor[0]] = new_state_fvalue
         
    return False #return False upon no solution found

#======================================================================
# Below are my 3 helper functions
#======================================================================
def cost(route):
    """
    A helper function that counts and return the total cost of the given route from the start state to the given state
    NOTE: every element in route has the form ((state),action,cost), so retrieving element[-1] will do.
    
    route: a sequence of ((state),action,cost)
    """
    cost = 0;
    for states in route:
        cost += states[-1]
    return cost

def retrieve_direction(route):
    """
    A helper funciton for BFS which takes the given solution route in form of 
    ((state, action, cost)...) and retrieves needed direction info and
    returns a list of actions to perform, using the retrieve_solution funciton below.
    
    route: a sequence of ((state),action,cost)
    """
    directions = []
    
    for node in route: # Here since I know the index of solution direction is the 
                       # second of the tuple, therefore retrieving index 1 will do. 
        directions.append(node[1])
    print directions
    return retrieve_solution(directions)    

def retrieve_solution(route):
    """
    A helper funciton which takes the given solution sequence in form of 
    [direction] and retrieves needed direction info and
    returns a list of actions to perform.
    
    route: a sequence of ((state),action,cost)
    """
    from game import Directions  
    
    solution = []
    
    for direction in route:
        if direction == 'South' :
            direction = Directions.SOUTH
        elif direction == 'North':
            direction = Directions.NORTH
        elif direction == 'East':
            direction = Directions.EAST
        elif direction == 'West':
            direction = Directions.WEST
        else:
            continue # Here I use continue because I know this 
                     # condition will only be met when getting action "None" from the initial state.
                     # And that no action is available to get to the initial state.
        solution.append(direction)
    print "SOLUTION",solution
    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
