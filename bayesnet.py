import networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import string

class Distrib:
  # class to store discrete distributions
  def __init__(self, arr, vars):
    self.arr = np.array(arr)
    self.vars = vars
    #self.check_sum1()
    self.check_nvars()

  def check_sum1(self):
    assert np.all(np.sum(self.arr, axis=0)==1.), "Some values don't sum up to 1"
  def check_nvars(self):
    assert self.arr.ndim == len(self.vars), "arr.ndim {} and len(self.vars) {}".format(self.arr.ndim, len(self.vars))
    
  def get_arr(self):
    return self.arr
  def get_vars(self):
    return self.vars

  def __str__(self):
    return "Distrib of vars ", str(self.vars), "\n and arr ", str(self.arr)
  def __repr__(self):
    return "Distrib of vars ", str(self.vars), "\n and arr ", str(self.arr)

class Node:
  """
  Stores a node, which is associated with a binary variable and a
  probability distribution.

  We consider nodes as falling into three general types: nodes that we
  observe, nodes of interest whose distributions we want to estimate, or
  nodes that are neither, which have to be marginalized over.
  """
  def __init__(self, distrib, parents=[], name=""):
    self.distrib = distrib
    self.name = name
    assert [parent.get_name() for parent in parents]==distrib.get_vars()[1:]
    self.parents = parents
  def get_parents(self): return self.parents
  def get_name(self): return self.name
  def get_distrib(self): return self.distrib

class BayesNet:
  # The central class that accomplishes the bayesian inference and plots
  def __init__(self, name):
    self.name = name
    self.nodes = []
    self.edges = []
    self.graph = nx.DiGraph()

  def add_nodes(self,nodes):
    # Add nodes to the graph
    self.nodes.extend(nodes)
    self.graph.add_nodes_from([n.name for n in nodes])

  def infer_edges(self):
    # Infer the edges from the probability distributions
    for n in self.nodes:
      for parent in n.parents:
        self.graph.add_edge(parent.name, n.name)
  
  def BFS_ordering(self):
    """
    Create an ordering of a graph in a Breadth First search fashion. No idea if it will perform well. 
    """
    self.order = []
    self.orderdict = {}
    raise NotImplementedError

  def heuristic_ordering(self):
    """
    Create an ordering of a graph that is most efficient.
    On page 12 of original paper. References: Dechter, 1992; Becker and Geiger, 1996).
    """
    raise NotImplementedError

  def plot_graph(self, pos=None):
    # Plot the graph and display what edges are connected
    if pos is None:
      pos = nx.spring_layout(self.graph)
    nx.draw(self.graph, pos, with_labels=True, node_size=1000, node_color='w')

  def bucket_elim(self, observations, interest_nodes, ordering, DEBUG=False):
    """
    Belief Propagation Bucket Elimination: 
    Given a graph, observations, a list of nodes of interest and an ordering,
    it calculates the posterior distribution of the first node in the ordering,
    and a MAP estimate for all nodes of interest. 
    """

    # do all of the observed nodes first. then unobserved

    buckets = self.bucket_init(observations, interest_nodes, ordering)
    #print("buckets ", buckets)
    res_BEL, int_MAP = self.bucket_backward(observations, interest_nodes, ordering, buckets, DEBUG=DEBUG)

    res_MAP = self.bucket_forward(observations, interest_nodes, ordering, buckets, int_MAP, DEBUG=DEBUG)

    return res_BEL, res_MAP

  def bucket_init(self, observations, interest_nodes, ordering):
    """
    Assign conditional probabilities to buckets
    """
    # Define a function to be used in conjuction with the max operator to sort according to the ordering.
    d_name2int = {o:i for i,o in enumerate(ordering)}
    def highest_order(x):
      return d_name2int[x]

    # initialize buckets to nothing
    buckets = {}
    for noden in ordering: buckets[noden] = []

    # Assign all distributions to the largest bucket that it is related to. 
    for node in self.nodes:
      distrib = node.get_distrib()
      max_order = max(distrib.get_vars(), key=highest_order)
      buckets[max_order].append(distrib)

    return buckets

  def bucket_backward(self, observations, interest_nodes, ordering, buckets, DEBUG=False):
    """
    The Backwards step of bucket elimination propagates knowledge up the ordering.
    If only independent interest_nodes is chosen, this algorithm runs elim-bel for that interest_node
    If not only intependent interest_nodes is chosen, this algorithm sets up the calculation to find the MAP estimate in the forwards pass. 
    """
    if DEBUG: print("\n\n")
    print("Running bucket_backward")
    if DEBUG: print("\n")
    # Define a function to be used in conjuction with the max operator to sort according to the ordering.
    d_name2int = {o:i for i,o in enumerate(ordering)}
    def highest_order(x):
      return d_name2int[x]
    
    # Define a list containing the letters of the alphabet for einstein notation
    alphabet = list(string.ascii_lowercase) 

    int_MAP = {} # the intermediate results for the MAP calculation, to be finalized in the bucket_forward step
    res_BEL = {} # the final result of the elim-bel algorithm that calculates the posterior for the interest_node

    for noden in ordering[::-1]: #loop from high ordering to low ordering, noden= nodename
      # Depending on whether we have data for this node and whether the node is of interest the algorithm will change.
      MODE = "SUM" # SUM for belief estimation, and MAX for MAP for nodes of interest, OBS for observation
      if noden in interest_nodes: MODE= "MAX"
      if noden in observations: MODE= "OBS"
      if DEBUG: print("Working on ", noden)

      if MODE=="SUM" and len(buckets[noden])==1:
        # Given a belief network and a topological ordering X1; :::;Xn,
        # algorithm elim-bel can skip a bucket if at the time of processing, the bucket
        # contains no evidence variable, no query variable and no newly computed
        # function.
        continue
      
      if MODE =="OBS":
        # "It would be more eective however, to apply the assignment b = 1
        # to each function in a bucket separately and then put the resulting func
        # tions into lower buckets." 
        
        for distrib in buckets[noden]:
          arr = distrib.get_arr()
          vars = distrib.get_vars()
          i_obs = observations[noden]
          # Find the axis in which noden occurs. 
          iaxis = next(ivar for ivar, avar in enumerate(vars) if avar == noden) # finds first true statement
          # Assign the value to the column of the observation and discard other columns
          arr = np.take(arr, i_obs, axis=iaxis)

          if len(vars)>1: # if this assignment affects other buckets, propagate
            max_order = max(vars[1:], key=highest_order)
            buckets[max_order].append( Distrib(arr, vars[1:]) )
        continue
       
      # Find all the dimensions (=variables) that affect this bucket
      dims = set()
      for distrib in buckets[noden]:
        dims.update(distrib.get_vars())
        if DEBUG: print(" variables in distributions", distrib.get_vars()) 
      dims.remove(noden)
      dims = [noden] + list(dims)
      # Assign each dimension a value from the alphabet. a=noden
      dim_dict = {ndim: alphabet[idim]  for idim, ndim in enumerate(dims)}

      # Let us use Einstein notation to figure out how to multiply the matrices.
      einstein_string_parts = []
      for distrib in buckets[noden]:
        einstein_string_parts.append("".join([dim_dict[nvar] for nvar in distrib.get_vars()]))
      einstein_string = ",".join(einstein_string_parts) + "->" + "".join(alphabet[:len(dims)])
      if DEBUG: print("einstein_string ", einstein_string)

      # Run the Sum via Einstein notation
      arr = np.einsum(einstein_string, *[distrib.get_arr() for distrib in buckets[noden]])
      #print("arr before ", arr)
      #print("MODE ", MODE )

      if MODE== "SUM": # Sum over the axis of noden 
        arr = np.sum(arr, axis=0)
      elif MODE=="MAX": # Take the maximum over the axis of noden
        if len(dims)==1: 
          # If this node is independent of all other nodes higher in the ordering, then compute its posterior
          if DEBUG: print("Saving the posterior for {} to res_BEL".format(noden)) 
          res_BEL[noden] = arr*1./sum(arr)
        imax = np.argmax(arr, axis=0)
        if DEBUG: print("Saving the maximum for {} to int_MAP".format(noden))
        int_MAP[noden] = Distrib(imax, dims[1:])
        arr = np.amax(arr, axis=0)
      elif MODE=="OBS": # Assign the observed value over the axis of noden 
        i_obs = observations[noden]
        arr = np.take(arr, i_obs, axis=0)
      #print("arr after ", arr)

      if len(dims)>1: # if this assignment affects other buckets, propagate
        max_order = max(dims[1:], key=highest_order)
        buckets[max_order].append( Distrib(arr, dims[1:]) )
        
    return res_BEL, int_MAP

  def bucket_forward(self, observations, interest_nodes, ordering, buckets, int_MAP, DEBUG=False):
    """
    The Forward step of bucket elimination computes the MAP assignment after precomputation in the Backward step.
    """
    if DEBUG: print("\n")
    print("Running bucket_forward")
    if DEBUG: print("\n")
    res_MAP = {}
    for noden in ordering:
      if noden in interest_nodes:
        arr = int_MAP[noden].get_arr()
        vars = int_MAP[noden].get_vars()
        for var in vars: # Plug in maxes for already resolved variables
          arr = arr[res_MAP[var],] 
        res_MAP[noden] = int(arr)
    return res_MAP






def generate_distribution(vars):
  """
  Given a list of variable names, generate a valid Distrib with
  random probabilities.
  """
  # TODO(afra): implement
  random_dist = np.random.dirichlet(np.ones(2),size=1*1).flatten()
  distrib = Distrib(random_dist, vars=[vars])
  return distrib

def generate_simple_problem_net():
  """
  Returns a hard-coded Bayes Net.
  """
  # Attack1 is at the top of the tree and has no parents
  # There are two possible states. one with prior of 1/5 and and with a prior of 4/5
  attack1_d = Distrib([1./5, 4./5], vars=["attack1"])
  attack1 = Node(attack1_d, name="attack1")

  # Attack2 is at the top of the tree and has no parents
  attack2_d = Distrib([1./3, 2./3], vars=["attack2"])
  attack2 = Node(attack2_d, name="attack2")

  # Server1 has parents attack1 and attack2
  # Its conditional propability distribution is captured by an array. 
  # The vars argument gives a name to the different indices. 
  # The first argument should be the nodename and followed by its parents. 
  server1_d = Distrib([[[0.9,0.4], [0.5,0.3]], [[0.1,0.6], [0.5,0.7]]], vars=["server1", "attack1", "attack2"])
  server1_d.check_sum1() #the sum over axis 0 should add up to 1, as it is a conditional probability
  server1 = Node(server1_d, parents=[attack1, attack2], name="server1")

  # TODO(afra): Reimplement the following CondProbTable, which is not
  # a class that we apparently ever implemented, as a Distrib:
  #  server1 = CondProbTable(
  #      [[ 'Y', 'Y', 'Y', 1.0 ],
  #        [ 'Y', 'Y', 'N', 0.0 ],
  #        [ 'Y', 'N', 'Y', 1.0 ],
  #        [ 'Y', 'N', 'N', 0.0 ],
  #        [ 'N', 'Y', 'Y', 0.5 ],
  #        [ 'N', 'Y', 'N', 0.5 ],
  #        [ 'N', 'N', 'Y', 0.0 ],
  #        [ 'N', 'N', 'N', 1.0 ]], [attack1, attack2])

  # Create the Bayesian network object with a useful name
  model = BayesNet("Simple Attack Estimation")

  # Add the three states to the network 
  model.add_nodes([attack1, attack2, server1])
  model.infer_edges()

  return model


def generate_problem():
  """
  Returns  hard-coded Bayes Net nodes
  """
  attack1 = Node(Distrib([4./5, 1./5], vars=["attack1"]), name="attack1")
  attack2 = Node(Distrib([2./3, 1./3], vars=["attack2"]), name="attack2")
  attack3 = Node(Distrib([3./4, 1./4], vars=["attack3"]), name="attack3")

  subsys1_d = Distrib([[[0.1,0.99], [0.5,0.3]], [[0.9,0.01], [0.5,0.7]]],
                    vars=["subsys1", "attack1", "attack2"])
  subsys1_d.check_sum1()
  subsys1 = Node(subsys1_d, parents=[attack1, attack2], name="subsys1")

  subsys2_d = Distrib([[[0.7,0.5], [0.1,0.8]], [[0.3,0.5], [0.9,0.2]]],
                    vars=["subsys1", "attack1", "attack3"])
  subsys2_d.check_sum1()
  subsys2 = Node(subsys2_d, parents=[attack1, attack3], name="subsys2")

  ws1_d = Distrib([[0.999], [0.001]], vars=["ws1", "subsys1"])
  ws1_d.check_sum1()
  ws1 = Node(ws1_d, parents=[subsys1], name="ws1")

  ws2_d = Distrib([[[0.9,0.8], [0.5,0.3]], [[0.1,0.2], [0.5,0.7]]],
                vars=["ws2", "subsys1", "subsys2"])
  ws2_d.check_sum1()
  ws2 = Node(ws2_d, parents=[subsys1, subsys2], name="ws2")

  ws3_d = Distrib([[[[0.1,0.99], [0.5,0.3]], [[0.7,0.5], [0.1,0.2]]],
                 [[[0.9,0.01], [0.5,0.7]], [[0.3,0.5], [0.9,0.8]]]],
                vars=["ws3", "subsys1", "attack2", "subsys2"])
  ws3_d.check_sum1()
  ws3 = Node(ws3_d, parents=[subsys1, attack2, subsys2], name="ws3")

  ws4_d = Distrib([[0.3], [0.7]], vars=["ws4", "subsys2"])
  ws4_d.check_sum1()
  ws4 = Node(ws4_d, parents=[subsys2], name="ws4")


  pos = {"attack1": [-1, 1], "attack2": [0, 1], "attack3": [1, 1],
       "subsys1": [-2, 0], "subsys2": [2, 0],
       "ws1": [-2, -1], "ws2": [-1, -1], "ws3": [0, -1], "ws4": [2, -1]}

  return ([attack1, attack2, attack3, subsys1, subsys2, ws1, ws2, ws3, ws4], pos)

def simulate_observations(attack_var_assignments, model):
  """
  Given a model and a dictionary of assignments to the attack variables,
  simulates what the resulting observations should look like.
  Returns a dictionary of observations.
  """
  # TODO(afra): implement
  raise NotImplementedError

def simple_attack(model):
  """
  Simulates a simple attack by setting a single attack variable to True and
  generating the resulting observations
  """
  # TODO(afra): implement
  raise NotImplementedError

def multiple_attack(model, n=3):
  """
  Simulates an attack by setting multiple attack variables to True and
  generating the resulting observations
  n: The number of attack variables
  """
  # TODO(afra): implement
  raise NotImplementedError

def generate_attack(model, mode="simple"):
  """
  Generate some observations given one or many attack variables being true.
  Our algorithm then tries to recover these.
  Returns a dictionary of observations and a list of attacks that did happen.
  """
  if mode=="simple": return simple_attack(model)
  if mode=="multiple": return multiple_attack(model)

def display_bucket_elimination_procedure():
  """
  Show the bucket elimination procedure for all nodes in the graph and compute
  the induced width
  """
  raise NotImplementedError

def display_posterior_distribution():
  """
  Plot the model graph annotated with posterior distributions for all the
  attack variables.
  """
  raise NotImplementedError

def display_most_likely_configuration():
  """
  Plot the model graph annotated with the most probable configuration for each
  attack variable
  """
  raise NotImplementedError



