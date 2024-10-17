import matplotlib.pyplot as plt
import random
import numpy as np
from dataclasses import dataclass
import torch

class TownEnvironment:
    defaultObstacles =[]
    #[(1,1),(2,1),(3,1),(4,1),(1,2),(4,2),(2,3),(2,4),(0,5),(5,5),(5,6),(5,7),(4,7),(3,7),(7,4),(8,4),(8,5)]
    agentsid = {}
    agentsid_reverse = {}
    @dataclass
    class Agent:
        id: int
        pos: tuple
        agenttypeid:int
        agenttype:str

        def get_action(self,state, env):
            # Placeholder for action selection logic
            actions = [(0,1), (0,-1), (-1,0), (1,0),(0,0)]
            return actions
        
        def update(self,state, env, action):
            
            x,y = env.actions_lookup(action)
            dx,dy = self.pos
            self.pos = (x+dx,y+dy)
            
            env.grid[x][y] = 0
            env.grid[x+dx][y+dy]=2

    def __init__(self, size=10, obstacles=None):
        self.size = size
        self.grid = np.zeros((self.size, self.size))
        self.obstacles = obstacles if obstacles else self.defaultObstacles
        self.updateobstacles()
        self.agents = {
            'police': [],
            #self.Agent(1, (0, 0))
            #self.Agent(2, (9, 9))
            'thief': []
        }

    def resetagent(self,agent,pos=None):
        if pos is not None:
            x,y=agent.pos
            self.grid[x][y]=0
            agent.pos = (0,0)
            self.grid[0][0]=agent.agenttypeid
            return 
        positions = np.argwhere(self.grid == agent.agenttypeid)
        for position in positions:
            self.grid[tuple(position)]=0
        for y in self.agents[agent.agenttype]:
            self.assign_reset_agents(self.getzero_block(),y)

    def assign_reset_agents(self,pos,agent):
        agent.pos = pos
        #print(pos)
        x,y = pos
        self.grid[x][y]=agent.agenttypeid

    def reset(self):
        self.grid =  np.zeros((10, 10))
        self.obstacles = self.defaultObstacles
        self.updateobstacles()
        for x in self.agentsid:
            for y in self.agents[x]:
                self.assign_reset_agents(self.getzero_block(),y)
        return self.grid

    def agent_id_lookup(self,agent_type):
        if agent_type not in self.agentsid:
            self.agentsid[agent_type] = len(self.agentsid)+2
        return self.agentsid[agent_type]

    def updateobstacles(self):
        for (x,y) in self.obstacles:
            self.grid[x,y] = 1
    
    def next_state(self):
        return self.grid

    def add_agent(self, agent_type,pos):
        agentypeid = self.agent_id_lookup(agent_type)
        if len(self.agents[agent_type]) == 0:
            max_agent_id = 1
        else:
            max_agent_id = max(agent.id for agent in self.agents[agent_type])
        new_agent = self.Agent(max_agent_id + 1, pos,agentypeid,agent_type)
        self.agents[agent_type].append(new_agent)
        self.grid[pos[0]][pos[1]]=self.agentsid[agent_type]
        print(self.grid)
        return new_agent
    
    def getzero_block(self):
        zeros = [(i, j) for i in range(len(self.grid)) for j in range(len(self.grid[i])) if self.grid[i][j] == 0]
        pos = random.choice(zeros) if zeros else None
        if pos is not None:
            return pos
        else:
            return None

    def get_state(self):
        # Placeholder for state representation
        return self.grid
    
    def actions_lookup(self,val):
        if val == 1:
            return (0,1)
        elif val == 2:
            return (0,-1)
        elif val==3:
            return (1,0)
        elif val == 4:
            return (-1,0)
        elif val == 5:
            return (0,0)


    def step(self, agent, targetagent, action):
        # Convert the agent's position to a list for easier manipulation
        dx,dy = agent.pos
        x,y = action
        
        x=x+dx
        y=y+dy
        done = False
        # Check if the new position is valid
        if self.is_valid_position((x, y),targetagent.agenttypeid):
            if(self.grid[x][y]==targetagent.agenttypeid):
                return self.get_state(),100, True
            else:
                agent.pos = (x, y)
                self.grid[dx][dy]=0
                self.grid[x][y]=agent.agenttypeid
                if dx-x ==0 and dy-y == 0:
                    reward = -0.05
                else:
                    reward = -0.1
            return self.get_state(),reward, done
        else:
            return self.get_state(),-1, False  # Penalty for invalid move, not done

    def is_valid_position(self, pos,val):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and (self.grid[x][y] == val or self.grid[x, y] == 0)

    def get_reward(self, agent):
        # Placeholder for reward calculation
        return -1

    def is_done(self, agent1,agent2):
        # Placeholder for done condition
        if agent1.pos == agent2.pos:
            return True
        return False
    
    def checkaction(self,val,pos):
        actions = []
        for action in val:
            x,y = pos
            dx,dy = action
            if self.is_valid_position((x+dx,y+dy)):
                actions.append(action)
        return actions
    
    def resetagentpos(self, agent, pos):
        self.grid[agent.pos[0]][agent.pos[1]]=0
        agent.pos = pos
        self.grid[pos[0]][pos[1]]=agent.agenttypeid

    def render(self):
        env=self.get_state().copy()
        plt.imshow(env, cmap='coolwarm')
        plt.show(block=False)
        plt.pause(0.02)
        plt.clf()
