from glob import glob
import json

files = sorted(glob('ReplayData/*'))

class Tick:
    def __init__(self, TickIndex) -> None:
        self.TickIndex = TickIndex
    
    @staticmethod
    def from_string(str):
        idx, str = str.split(' ',1)
        str = str.replace(': ', ':').replace(', ', ',').split()
        
        
        retval = Tick(int(idx.strip().strip(':')))
        
        
        check = [
            'PacketData:MovingData',
            'PacketData:AnimationData',
            'PacketData:null',
            'PacketData:ChatData',
            'PacketData:SpawnData',
            'PacketData:InvData',
            'PacketData:BlockChangeData',
            'INCLUDED!!'
            ]
        type_marker = [x for x in check if x in str][0].split(':')
        if type_marker[0] == 'INCLUDED!!':            
            retval.__dict__['PacketData'] = 'INCLUDED!!'
            # return retval
        else:
            retval.__dict__[type_marker[0]] = type_marker[1]
        # except Exception as e:
        #     print('exception')
        #     print(type_marker,str)
        
        # retval.__dict__['data'] = str
        
        
        
        if retval.PacketData == 'MovingData':
            x = [x for x in str if '(x,y,z)' in x][0].split(':')
            retval.__dict__['xyz'] = eval(x[1])
            x = [x for x in str if '(yaw,pitch)' in x][0].split(':')
            retval.__dict__['orientation'] = eval(x[1])
            x = [x for x in str if 'Name' in x][0].split(':')
            retval.__dict__['Name'] = x[1]
        elif retval.PacketData == 'AnimationData':
            x = [x for x in str if 'Name' in x][0].split(':')
            retval.__dict__['Name'] = x[1]
        elif retval.PacketData == 'null':
            pass
        elif retval.PacketData == 'ChatData':
            retval.__dict__['message'] = ' '.join(str).split('message:')[1]
            x = [x for x in str if 'Name' in x][0].split(':')
            retval.__dict__['Name'] = x[1]
        elif retval.PacketData == 'SpawnData':
            x = [x for x in str if '(x,y,z)' in x][0].split(':')
            retval.__dict__['xyz'] = eval(x[1])
            x = [x for x in str if '(yaw,pitch)' in x][0].split(':')
            retval.__dict__['orientation'] = eval(x[1])
            x = [x for x in str if 'Name' in x][0].split(':')
            retval.__dict__['Name'] = x[1]
        elif retval.PacketData == 'InvData':
            x = [x for x in str if 'items' in x]            
            retval.__dict__['items'] = [y.split('type=')[1].strip('}') for y in x]
            x = [x for x in str if 'Name' in x][0].split(':')
            retval.__dict__['Name'] = x[1]
        elif retval.PacketData == 'BlockChangeData':
            x = [x for x in str if '(x,y,z)' in x][0].split(':')
            retval.__dict__['xyz'] = eval(x[1])
            x = [x for x in str if '(yaw,pitch)' in x][0].split(':')
            retval.__dict__['orientation'] = eval(x[1])
            x = [x for x in str if 'items' in x]
            retval.__dict__['items'] = [y.split('type=')[1].strip('}') for y in x]
            x = [x for x in str if 'Name' in x][0].split(':')
            retval.__dict__['Name'] = x[1]
            # print(retval)
            # print(str)
            # exit(0)
        elif type_marker[0] == 'INCLUDED!!':            
            retval.__dict__['PacketData'] = 'None'
        else:
            print('else')
            print(retval)
            print(str)
            exit(0)
        
        
        # retval = Tick(int(idx.strip().strip(':')),str)
        

        return retval
    
    def __repr__(self) -> str:
        return str(list(self.__dict__.items()))

def proc_tick(tick_str):
    return Tick.from_string(tick_str)

def proc_action(action):
    ticks = action.split('TickIndex')
    
    return list(map(proc_tick,ticks[1:]))