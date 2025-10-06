import heapq

#FOR STORING ALL DIFFERENT DATATYPES IN ONE
class State:
    def __init__(self,state,parent=None,g=0):
        self.parent=parent
        self.state=state
        self.g=g
    def __lt__ (self,other):
        return self.g<other.g

#CHECKS FOR THE VALID TRANSITION
def isValid(parent,next_state):
    e=set()
    w=set()
    c=0
    for i in parent:
        if i=='E': e.add(c)
        elif i=='W': w.add(c)
        c=c+1
    check=-1
    c=0
    for i in next_state:
        if i=='E' :
            if (c in e):
                e.remove(c)
            else : check=c
        elif i=='W':
            if c in w:
                w.remove(c)
            else: check=c
        c=c+1
    
    if e:
        if min(e)>check: return False
    
    else:
        if min(w)<check : return False

    
    return True


#GIVES ALL THE NEXT POSSIBLE TRANSITIONS
def get_successor(state):
    idx=state.index('0')
    moves=[1,-1,2,-2]
    successor=[]
    for m in moves:
        next_state=list(state)
        im=idx+m
        
        if im >=0 and im<7:
            next_state[im],next_state[idx]=next_state[idx],next_state[im]
            if(isValid(state,next_state)):
                successor.append(next_state)
        
    return successor

#FOR SEEING ALL THE POSSIBLE PATHS FROM INITIAL TO FINAL STATES AND RETURNS THE FIRST PATH THAT REACHES THE GOAL.
def bfs(start,goal):
    queue=[]
    heapq.heappush(queue,(start.g,start))

    vis=set()
    while(queue):
        _,state=heapq.heappop(queue)

        if tuple(state.state) in vis:
            continue
        if tuple(state.state)==tuple(goal):
            path=[]
            while state:
                path.append(state)
                state=state.parent
            return path[::-1],vis.__len__()
        
        vis.add(tuple(state.state))
        list=tuple(state.state)
        successors=get_successor(list)

        for s in successors:
            st=State(s,state,state.g+1)
            heapq.heappush(queue,(st.g,st))
    return

stlist=['E','E','E','0','W','W','W']
goallist=['W','W','W','0','E','E','E']
start=State(stlist,None,0)

path, vis =bfs(start,goallist)

for p in path:
    print (p.state)

print(f"total number of nodes visited are: {vis}")
print(f"Number of nodes in this path are : {path.__len__()}")
