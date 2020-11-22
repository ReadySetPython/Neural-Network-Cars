import pygame
import random
import os
import math
import numpy as np
import time
import sys
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image
from operator import attrgetter

pygame.init() #Initialize pygame
#Some variables initializations
img = 0 #This one is used when recording frames
size = width,height = 1600, 900 #Size to use when creating pygame window

#Colors
white = (255,255,255)
green = (0, 255, 0) 
blue = (0, 0, 128)  
black = (0,0,0)
gray = pygame.Color('gray12')
Color_line = (255,0,0)

generation = 1
mutationRate = 90
FPS = 30
selectedCars = []
selected = 0
lines = True #If true then lines of player are shown
player = True #If true then player is shown
display_info = True #If true then display info is shown
frames = 0
maxspeed = 10 
number_track = 1

white_small_car = pygame.image.load('Images\Sprites\white_small.png')
white_big_car = pygame.image.load('Images\Sprites\white_big.png')
green_small_car = pygame.image.load('Images\Sprites\green_small.png')
green_big_car = pygame.image.load('Images\Sprites\green_big.png')

bg = pygame.image.load('bg7.png')
bg4 = pygame.image.load('bg4.png')


def calculateDistance(x1,y1,x2,y2): #Used to calculate distance between points
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def rotation(origin, point, angle): #Used to rotate points #rotate(origin, point, math.radians(10))
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
    
def move(point, angle, unit): #Translate a point in a given direction
  x = point[0]
  y = point[1]
  rad = math.radians(-angle % 360)

  x += unit*math.sin(rad)
  y += unit*math.cos(rad)

  return x, y
  
def sigmoid(z): #Sigmoid function, used as the neurons activation function
    return 1.0/(1.0+np.exp(-z))

def mutateOneWeightGene(parent1, child1):
    sizenn = len(child1.sizes)
    
    #Copy parent weights into child weights
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]          
                
    #Copy parent biases into child biases
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]

    genomeWeights = [] #This will be a list containing all weights, easier to modify this way
    
    for i in range(sizenn-1): #i=0,1
        for j in range(child1.sizes[i]*child1.sizes[i+1]):
            genomeWeights.append(child1.weights[i].item(j))
    
    #Modify a random gene by a random amount
    r1 = random.randint(0, len(genomeWeights)-1)
    genomeWeights[r1] = genomeWeights[r1]*random.uniform(0.8, 1.2)
    
    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genomeWeights[count]
                count += 1
    return

def mutateOneBiasesGene(parent1, child1):
    sizenn = len(child1.sizes)
    

    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]          
                
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]

    genomeBiases = []
    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            genomeBiases.append(child1.biases[i].item(j))
            
    r1 = random.randint(0, len(genomeBiases)-1)
    genomeBiases[r1] = genomeBiases[r1]*random.uniform(0.8, 1.2)
    
    count = 0    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = genomeBiases[count]
                count += 1
    return

def uniformCrossOverWeights(parent1, parent2, child1, child2): #Given two parent car objects, it modifies the children car objects weights
    sizenn = len(child1.sizes) #3 si car1=Car([2, 4, 3])
    
    #Copy parent weights into child weights
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]
                
    #Copy parent biases into child biases
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = parent2.biases[i][j]

    genome1 = [] #This will be a list containing all weights of child1
    genome2 = [] #This will be a list containing all weights of child2
        
    for i in range(sizenn-1): #i=0,1
        for j in range(child1.sizes[i]*child1.sizes[i+1]):
            genome1.append(child1.weights[i].item(j))
            
    for i in range(sizenn-1): #i=0,1
        for j in range(child2.sizes[i]*child2.sizes[i+1]):
            genome2.append(child2.weights[i].item(j))

    #Crossover weights          
    alter = True    
    for i in range(len(genome1)):
        if alter == True:
            aux = genome1[i]
            genome1[i] = genome2[i]
            genome2[i] = aux
            alter = False
        else:
            alter = True

    #Go back from genome list to weights numpy array on child object
    count = 0
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genome1[count]
                count += 1
          
    count = 0
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            for k in range(child2.sizes[i]):
                child2.weights[i][j][k] = genome2[count]
                count += 1              
    return 

def uniformCrossOverBiases(parent1, parent2, child1, child2): #Given two parent car objects, it modifies the children car objects biases
    sizenn = len(parent1.sizes)
    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child1.biases[i][j] = parent1.biases[i][j]
                
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = parent2.biases[i][j]
                
    genome1 = []
    genome2 = []
    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
            genome1.append(child1.biases[i].item(j))
            
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
            genome2.append(child2.biases[i].item(j))
     
    alter = True    
    for i in range(len(genome1)):
        if alter == True:
            aux = genome1[i]
            genome1[i] = genome2[i]
            genome2[i] = aux
            alter = False
        else:
            alter = True
    
    count = 0    
    for i in range(sizenn-1):
        for j in range(child1.sizes[i+1]):
                child1.biases[i][j] = genome1[count]
                count += 1
                
    count = 0    
    for i in range(sizenn-1):
        for j in range(child2.sizes[i+1]):
                child2.biases[i][j] = genome2[count]
                count += 1
    return 

def generateRandomMap(screen):
    SCREEN = screen
    
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 128)
    
    WINDOW_HEIGHT = 730
    WINDOW_WIDTH = 1460 #These are for the maze/grid, not for the pygame window size

    blockSize = 146 #Set the size of the grid block
    rows, cols = (int(WINDOW_WIDTH/blockSize), int(WINDOW_HEIGHT/blockSize))  
    maze = Maze(rows,cols,0,0)
    
    trackLenght = 1
    movex = 70
    movey = 85
    
    #Choose a start cell
    startx, starty = 0, 3
    currentCell = maze.cell_at(startx, starty)
    
    #Load track images!
    straight1 = pygame.image.load('Images\TracksMapGen\Straight1.png')
    straight1Rect = straight1.get_rect()

    straight2 = pygame.image.load('Images\TracksMapGen\Straight2.png')
    straight2Rect = straight2.get_rect()

    curve1 = pygame.image.load('Images\TracksMapGen\Curve1.png')
    curve1Rect = curve1.get_rect()

    curve2 = pygame.image.load('Images\TracksMapGen\Curve2.png')
    curve2Rect = curve2.get_rect()

    curve3 = pygame.image.load('Images\TracksMapGen\Curve3.png')
    curve3Rect = curve3.get_rect()

    curve4 = pygame.image.load('Images\TracksMapGen\Curve4.png')
    curve4Rect = curve4.get_rect()

    straight1Top = pygame.image.load('Images\TracksMapGen\Straight1Top.png')
    straight1RectTop = straight1Top.get_rect()

    straight2Top = pygame.image.load('Images\TracksMapGen\Straight2Top.png')
    straight2RectTop = straight2Top.get_rect()

    curve1Top = pygame.image.load('Images\TracksMapGen\Curve1Top.png')
    curve1RectTop = curve1Top.get_rect()

    curve2Top = pygame.image.load('Images\TracksMapGen\Curve2Top.png')
    curve2RectTop = curve2Top.get_rect()

    curve3Top = pygame.image.load('Images\TracksMapGen\Curve3Top.png')
    curve3RectTop = curve3Top.get_rect()

    curve4Top = pygame.image.load('Images\TracksMapGen\Curve4Top.png')
    curve4RectTop = curve4Top.get_rect()
    
    initialTop = pygame.image.load('Images\TracksMapGen\Initial.png')
    initialRectTop = initialTop.get_rect()

    bg = pygame.image.load('Images\TracksMapGen\Background.png')

    while True:

        #CurrentCell is the one at (0,3) position, im gonna look if there are unvisited cells from there
        if len(maze.find_valid_neighbours(currentCell)) > 0:
            if currentCell.x == 0 and currentCell.y == 3: #Second cell is always the one on top of first cell bc first cell is always a straight vertical track
                oldCell = currentCell
                currentCell = maze.cell_at(oldCell.x,oldCell.y-1)
                currentCell.color = GREEN
                oldCell.knock_down_wall(currentCell, "N")
                trackLenght += 1 #Keep track of length so to discard very short generated tracks
            else:
                random_unvisited_direction = random.choice(maze.find_valid_neighbours(currentCell))[0] #Pick a random direction to move
                oldCell = currentCell
                if random_unvisited_direction == "N": #Move according to the direction picked
                    currentCell = maze.cell_at(oldCell.x,oldCell.y-1)
                elif random_unvisited_direction == "S":
                    currentCell = maze.cell_at(oldCell.x,oldCell.y+1)
                elif random_unvisited_direction == "E":
                    currentCell = maze.cell_at(oldCell.x+1,oldCell.y)
                elif random_unvisited_direction == "W":
                    currentCell = maze.cell_at(oldCell.x-1,oldCell.y)
                
                oldCell.knock_down_wall(currentCell, random_unvisited_direction)
                trackLenght += 1
            
        else:
            #Track is ready (back in initial position)! lets check if its long enough
            if currentCell.x == 0 and currentCell.y == 4 and trackLenght > 40:
                SCREEN.fill((0,0,0))
                currentCell.knock_down_wall(maze.cell_at(0,3), "N")
                
                for x in range(0, WINDOW_WIDTH, blockSize):
                    for y in range(0, WINDOW_HEIGHT, blockSize):
                        currentCell = maze.cell_at(int(x/blockSize),int(y/blockSize))
                        currentCell.color = (0,0,1,255)
                
                for x in range(0, WINDOW_WIDTH, blockSize):
                    for y in range(0, WINDOW_HEIGHT, blockSize):
                        currentCell = maze.cell_at(int(x/blockSize),int(y/blockSize))
                        
                        if currentCell.walls["N"] == False and currentCell.walls["S"] == False:
                            SCREEN.blit(straight2, straight2Rect.move(x+movex,y+movey))  
                        elif currentCell.walls["E"] == False and currentCell.walls["W"] == False:
                            SCREEN.blit(straight1, straight1Rect.move(x+movex,y+movey))  
                        elif currentCell.walls["N"] == False and currentCell.walls["W"] == False:
                            SCREEN.blit(curve3, curve3Rect.move(x+movex,y+movey)) 
                        elif currentCell.walls["W"] == False and currentCell.walls["S"] == False:
                            SCREEN.blit(curve2, curve2Rect.move(x+movex,y+movey))     
                        elif currentCell.walls["S"] == False and currentCell.walls["E"] == False:
                            SCREEN.blit(curve1, curve1Rect.move(x+movex,y+movey)) 
                        elif currentCell.walls["E"] == False and currentCell.walls["N"] == False:
                            SCREEN.blit(curve4, curve4Rect.move(x+movex,y+movey)) 
                
                #Save track and change background to transparent because that is how the main program needs the track image to be
                #You can leave the black background if you change the collision condition on the main program
                pygame.image.save(SCREEN, "randomGeneratedTrackBack.png")
                img = Image.open("randomGeneratedTrackBack.png")
                img = img.convert("RGBA")
                pixdata = img.load()
                for y in range(img.size[1]):
                    for x in range(img.size[0]):
                        if pixdata[x, y] == (0, 0, 0, 255) or pixdata[x, y] == (0, 0, 1, 255):
                            pixdata[x, y] = (0, 0, 0, 0)
                img.save("randomGeneratedTrackBack.png")

                SCREEN.blit(bg, (0,0))  
                for x in range(0, WINDOW_WIDTH, blockSize):
                    for y in range(0, WINDOW_HEIGHT, blockSize):
                        if x == 0 and y == 3*blockSize:
                            SCREEN.blit(initialTop, initialRectTop.move(x-20+movex,y+movey))
                        else:
                            currentCell = maze.cell_at(int(x/blockSize),int(y/blockSize))
                            if currentCell.walls["N"] == False and currentCell.walls["S"] == False:
                                SCREEN.blit(straight2Top, straight2RectTop.move(x-20+movex,y+movey))  
                            elif currentCell.walls["E"] == False and currentCell.walls["W"] == False:
                                SCREEN.blit(straight1Top, straight1RectTop.move(x+movex,y-20+movey))  
                            elif currentCell.walls["N"] == False and currentCell.walls["W"] == False:
                                SCREEN.blit(curve3Top, curve3RectTop.move(x-15+movex,y-15+movey)) 
                            elif currentCell.walls["W"] == False and currentCell.walls["S"] == False:
                                SCREEN.blit(curve2Top, curve2RectTop.move(x-15+movex,y-15+movey))     
                            elif currentCell.walls["E"] == False and currentCell.walls["N"] == False:
                                SCREEN.blit(curve4Top, curve4RectTop.move(x-15+movex,y-15+movey))
                            elif currentCell.walls["S"] == False and currentCell.walls["E"] == False:
                                SCREEN.blit(curve1Top, curve1RectTop.move(x-15+movex,y-15+movey)) 
                          
                pygame.image.save(SCREEN, "randomGeneratedTrackFront.png")

                
                break

                
            else:
                #It wasnt long enough so we start again
                trackLenght = 0
                for x in range(0, WINDOW_WIDTH, blockSize):
                    for y in range(0, WINDOW_HEIGHT, blockSize):
                        maze.cell_at(int(x/blockSize),int(y/blockSize)).walls["N"] = True
                        maze.cell_at(int(x/blockSize),int(y/blockSize)).walls["S"] = True
                        maze.cell_at(int(x/blockSize),int(y/blockSize)).walls["E"] = True
                        maze.cell_at(int(x/blockSize),int(y/blockSize)).walls["W"] = True
                        maze.cell_at(int(x/blockSize),int(y/blockSize)).color = 0, 0, 0
                
                #Force occupied cells
                maze.cell_at(3,3).walls["N"] = False
                maze.cell_at(4,3).walls["N"] = False
                maze.cell_at(5,3).walls["N"] = False
                maze.cell_at(6,3).walls["N"] = False

                currentCell = maze.cell_at(startx, starty)
    return

class Cell:
    #A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
        self.color = 0, 0, 0
        self.track = ""

    def has_all_walls(self):
        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        #Knock down the wall between cells self and other
        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

class Maze:
    #A Maze, represented as a grid of cells.
    def __init__(self, nx, ny, ix=0, iy=0):
        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]
    def cell_at(self, x, y):
        return self.maze_map[x][y]
    def find_valid_neighbours(self, cell):
        #Return a list of unvisited neighbours to cell.
        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

class Car:
  def __init__(self, sizes):
    self.score = 0
    self.num_layers = len(sizes) #Number of nn layers
    self.sizes = sizes #List with number of neurons per layer
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Biases
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #Weights 
    #c1, c2, c3, c4, c5 are five 2D points where the car could collided, updated in every frame
    self.c1 = 0,0
    self.c2 = 0,0
    self.c3 = 0,0
    self.c4 = 0,0
    self.c5 = 0,0
    #d1, d2, d3, d4, d5 are distances from the car to those points, updated every frame too and used as the input for the NN
    self.d1 = 0
    self.d2 = 0
    self.d3 = 0
    self.d4 = 0
    self.d5 = 0
    self.yaReste = False
    #The input and output of the NN must be in a numpy array format
    self.inp = np.array([[self.d1],[self.d2],[self.d3],[self.d4],[self.d5]])
    self.outp = np.array([[0],[0],[0],[0]])
    #Boolean used for toggling distance lines
    self.showlines = False
    #Initial location of the car
    self.x = 120
    self.y = 480
    self.center = self.x, self.y
    #Height and width of the car
    self.height = 35 #45
    self.width = 17 #25
    #These are the four corners of the car, using polygon instead of rectangle object, when rotating or moving the car, we rotate or move these
    self.d = self.x-(self.width/2),self.y-(self.height/2)
    self.c = self.x + self.width-(self.width/2), self.y-(self.height/2)
    self.b = self.x + self.width-(self.width/2), self.y + self.height-(self.height/2) #El rectangulo está centrado en (x,y)
    self.a = self.x-(self.width/2), self.y + self.height-(self.height/2)              #(a), (b), (c), (d) son los vertices
    #Velocity, acceleration and direction of the car
    self.velocity = 0
    self.acceleration = 0  
    self.angle = 180
    #Boolean which goes true when car collides
    self.collided = False
    #Car color and image
    self.color = white
    self.car_image = white_small_car
  def set_accel(self, accel): 
    self.acceleration = accel
  def rotate(self, rot): 
    self.angle += rot
    if self.angle > 360:
        self.angle = 0
    if self.angle < 0:
        self.angle = 360 + self.angle
  def update(self): #En cada frame actualizo los vertices (traslacion y rotacion) y los puntos de colision
    self.score += self.velocity
    if self.acceleration != 0:
        self.velocity += self.acceleration
        if self.velocity > maxspeed:
            self.velocity = maxspeed
        elif self.velocity < 0:
            self.velocity = 0
    else:
        self.velocity *= 0.92
        
    self.x, self.y = move((self.x, self.y), self.angle, self.velocity)
    self.center = self.x, self.y
    
    self.d = self.x-(self.width/2),self.y-(self.height/2)
    self.c = self.x + self.width-(self.width/2), self.y-(self.height/2)
    self.b = self.x + self.width-(self.width/2), self.y + self.height-(self.height/2) #El rectangulo está centrado en (x,y)
    self.a = self.x-(self.width/2), self.y + self.height-(self.height/2)              #(a), (b), (c), (d) son los vertices
        
    self.a = rotation((self.x,self.y), self.a, math.radians(self.angle)) 
    self.b = rotation((self.x,self.y), self.b, math.radians(self.angle))  
    self.c = rotation((self.x,self.y), self.c, math.radians(self.angle))  
    self.d = rotation((self.x,self.y), self.d, math.radians(self.angle))    
    
    self.c1 = move((self.x,self.y),self.angle,10)
    while bg4.get_at((int(self.c1[0]),int(self.c1[1]))).a!=0:
        self.c1 = move((self.c1[0],self.c1[1]),self.angle,10)
    while bg4.get_at((int(self.c1[0]),int(self.c1[1]))).a==0:
        self.c1 = move((self.c1[0],self.c1[1]),self.angle,-1)

    self.c2 = move((self.x,self.y),self.angle+45,10)
    while bg4.get_at((int(self.c2[0]),int(self.c2[1]))).a!=0:
        self.c2 = move((self.c2[0],self.c2[1]),self.angle+45,10)
    while bg4.get_at((int(self.c2[0]),int(self.c2[1]))).a==0:
        self.c2 = move((self.c2[0],self.c2[1]),self.angle+45,-1)

    self.c3 = move((self.x,self.y),self.angle-45,10)
    while bg4.get_at((int(self.c3[0]),int(self.c3[1]))).a!=0:
        self.c3 = move((self.c3[0],self.c3[1]),self.angle-45,10)
    while bg4.get_at((int(self.c3[0]),int(self.c3[1]))).a==0:
        self.c3 = move((self.c3[0],self.c3[1]),self.angle-45,-1)
        
    self.c4 = move((self.x,self.y),self.angle+90,10)
    while bg4.get_at((int(self.c4[0]),int(self.c4[1]))).a!=0:
        self.c4 = move((self.c4[0],self.c4[1]),self.angle+90,10)
    while bg4.get_at((int(self.c4[0]),int(self.c4[1]))).a==0:
        self.c4 = move((self.c4[0],self.c4[1]),self.angle+90,-1)
        
    self.c5 = move((self.x,self.y),self.angle-90,10)
    while bg4.get_at((int(self.c5[0]),int(self.c5[1]))).a!=0:
        self.c5 = move((self.c5[0],self.c5[1]),self.angle-90,10)
    while bg4.get_at((int(self.c5[0]),int(self.c5[1]))).a==0:
        self.c5 = move((self.c5[0],self.c5[1]),self.angle-90,-1)
        
    self.d1 = int(calculateDistance(self.center[0], self.center[1], self.c1[0], self.c1[1]))
    self.d2 = int(calculateDistance(self.center[0], self.center[1], self.c2[0], self.c2[1]))
    self.d3 = int(calculateDistance(self.center[0], self.center[1], self.c3[0], self.c3[1]))
    self.d4 = int(calculateDistance(self.center[0], self.center[1], self.c4[0], self.c4[1]))
    self.d5 = int(calculateDistance(self.center[0], self.center[1], self.c5[0], self.c5[1]))
    
    

  def draw(self,display):
    rotated_image = pygame.transform.rotate(self.car_image, -self.angle-180)
    rect_rotated_image = rotated_image.get_rect()
    rect_rotated_image.center = self.x, self.y
    gameDisplay.blit(rotated_image, rect_rotated_image)
  
    center = self.x, self.y
    if self.showlines: 
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c1,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c2,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c3,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c4,2)
        pygame.draw.line(gameDisplay,Color_line,(self.x,self.y),self.c5,2) 
    
  def showLines(self):
    self.showlines = not self.showlines
    
  def feedforward(self):
    #Return the output of the network
    self.inp = np.array([[self.d1],[self.d2],[self.d3],[self.d4],[self.d5],[self.velocity]])
    for b, w in zip(self.biases, self.weights):
        self.inp = sigmoid(np.dot(w, self.inp)+b)
    self.outp = self.inp
    return self.outp
  
  def collision(self):
      if (bg4.get_at((int(self.a[0]),int(self.a[1]))).a==0) or (bg4.get_at((int(self.b[0]),int(self.b[1]))).a==0) or (bg4.get_at((int(self.c[0]),int(self.c[1]))).a==0) or (bg4.get_at((int(self.d[0]),int(self.d[1]))).a==0):
        return True
      else:
        return False    

  def resetPosition(self):
      self.x = 120
      self.y = 480
      self.angle = 180
      return
      
  def takeAction(self): 
    if self.outp.item(0) > 0.5: #Accelerate
        self.set_accel(0.2)
    else:
        self.set_accel(0)      
    if self.outp.item(1) > 0.5: #Brake
        self.set_accel(-0.2)     
    if self.outp.item(2) > 0.5: #Turn right
        self.rotate(-5)    
    if self.outp.item(3) > 0.5: #Turn left
        self.rotate(5) 
    return
  
nnCars = [] #List of neural network cars 
num_of_nnCars = 50 #Number of neural network cars
alive = num_of_nnCars #Number of not collided (alive) cars
collidedCars = [] #List containing collided cars   

#These is just the text being displayed on pygame window
infoX = 1365
infoY = 600 
font = pygame.font.Font('freesansbold.ttf', 18) 
text1 = font.render('0..9 - Change Mutation', True, white) 
text2 = font.render('LMB - Select/Unselect', True, white)
text3 = font.render('RMB - Delete', True, white)
text4 = font.render('L - Show/Hide Lines', True, white)
text5 = font.render('R - Reset', True, white)
text6 = font.render('B - Breed', True, white)
text7 = font.render('C - Clean', True, white)
text8 = font.render('N - Next Track', True, white)
text9 = font.render('A - Toggle Player', True, white)
text10 = font.render('D - Toggle Info', True, white)
text11 = font.render('M - Breed and Next Track', True, white)
text1Rect = text1.get_rect().move(infoX,infoY)
text2Rect = text2.get_rect().move(infoX,infoY+text1Rect.height)
text3Rect = text3.get_rect().move(infoX,infoY+2*text1Rect.height)
text4Rect = text4.get_rect().move(infoX,infoY+3*text1Rect.height)
text5Rect = text5.get_rect().move(infoX,infoY+4*text1Rect.height)
text6Rect = text6.get_rect().move(infoX,infoY+5*text1Rect.height)
text7Rect = text7.get_rect().move(infoX,infoY+6*text1Rect.height)
text8Rect = text8.get_rect().move(infoX,infoY+7*text1Rect.height)
text9Rect = text9.get_rect().move(infoX,infoY+8*text1Rect.height)
text10Rect = text10.get_rect().move(infoX,infoY+9*text1Rect.height)
text11Rect = text11.get_rect().move(infoX,infoY+10*text1Rect.height)

def displayTexts():  
    infotextX = 20
    infotextY = 600
    infotext1 = font.render('Gen ' + str(generation), True, white) 
    infotext2 = font.render('Cars: ' + str(num_of_nnCars), True, white)
    infotext3 = font.render('Alive: ' + str(alive), True, white)
    infotext4 = font.render('Selected: ' + str(selected), True, white)
    if lines == True:
        infotext5 = font.render('Lines ON', True, white)
    else:
        infotext5 = font.render('Lines OFF', True, white)
    if player == True:
        infotext6 = font.render('Player ON', True, white)
    else:
        infotext6 = font.render('Player OFF', True, white)
    #infotext7 = font.render('Mutation: '+ str(2*mutationRate), True, white)
    #infotext8 = font.render('Frames: ' + str(frames), True, white)
    infotext9 = font.render('FPS: 30', True, white)
    infotext1Rect = infotext1.get_rect().move(infotextX,infotextY)
    infotext2Rect = infotext2.get_rect().move(infotextX,infotextY+infotext1Rect.height)
    infotext3Rect = infotext3.get_rect().move(infotextX,infotextY+2*infotext1Rect.height)
    infotext4Rect = infotext4.get_rect().move(infotextX,infotextY+3*infotext1Rect.height)
    infotext5Rect = infotext5.get_rect().move(infotextX,infotextY+4*infotext1Rect.height)
    infotext6Rect = infotext6.get_rect().move(infotextX,infotextY+5*infotext1Rect.height)
    #infotext7Rect = infotext7.get_rect().move(infotextX,infotextY+6*infotext1Rect.height)
    #infotext8Rect = infotext8.get_rect().move(infotextX,infotextY+7*infotext1Rect.height)
    infotext9Rect = infotext9.get_rect().move(infotextX,infotextY+6*infotext1Rect.height)

    gameDisplay.blit(text1, text1Rect)  
    gameDisplay.blit(text2, text2Rect)  
    gameDisplay.blit(text3, text3Rect) 
    gameDisplay.blit(text4, text4Rect) 
    gameDisplay.blit(text5, text5Rect) 
    gameDisplay.blit(text6, text6Rect)
    gameDisplay.blit(text7, text7Rect)   
    gameDisplay.blit(text8, text8Rect)  
    gameDisplay.blit(text9, text9Rect)     
    gameDisplay.blit(text10, text10Rect) 
    gameDisplay.blit(text11, text11Rect)  
    
    gameDisplay.blit(infotext1, infotext1Rect)  
    gameDisplay.blit(infotext2, infotext2Rect)  
    gameDisplay.blit(infotext3, infotext3Rect) 
    gameDisplay.blit(infotext4, infotext4Rect) 
    gameDisplay.blit(infotext5, infotext5Rect) 
    gameDisplay.blit(infotext6, infotext6Rect)
    #gameDisplay.blit(infotext7, infotext7Rect) 
    #gameDisplay.blit(infotext8, infotext8Rect) 
    gameDisplay.blit(infotext9, infotext9Rect) 
    return
 



gameDisplay = pygame.display.set_mode(size) #creates screen
clock = pygame.time.Clock()

inputLayer = 6
hiddenLayer = 6
outputLayer = 4
car = Car([inputLayer, hiddenLayer, outputLayer])
auxcar = Car([inputLayer, hiddenLayer, outputLayer])

for i in range(num_of_nnCars):
	nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))
   
def redrawGameWindow(): #Called on very frame   

    global alive  
    global frames
    global img
    
    frames += 1

    gameD = gameDisplay.blit(bg, (0,0))  
       
    #NN cars
    for nncar in nnCars:
        if not nncar.collided:
            nncar.update() #Update: Every car center coord, corners, directions, collision points and collision distances
        
        if nncar.collision(): #Check which car collided
            nncar.collided = True #If collided then change collided attribute to true
            if nncar.yaReste == False:
                alive -= 1
                nncar.yaReste = True

        else: #If not collided then feedforward the input and take an action
            nncar.feedforward()
            nncar.takeAction()
        nncar.draw(gameDisplay)
    

    #Same but for player
    if player:
        car.update()
        if car.collision():
            car.resetPosition()
            car.update()
        car.draw(gameDisplay)    
    if display_info:    
        displayTexts() 
    pygame.display.update() #updates the screen
    #Take a screenshot of every frame
    #pygame.image.save(gameDisplay, "pygameVideo/screenshot" + str(img) + ".jpeg")
    #img += 1
    
while True:
    #now1 = time.time()  
  
    for event in pygame.event.get(): #Check for events
        if event.type == pygame.QUIT:
            pygame.quit() #quits
            quit()
            
        if event.type == pygame.KEYDOWN: #If user uses the keyboard
            if event.key == ord ( "l" ): #If that key is l
                car.showLines()
                lines = not lines
            if event.key == ord ( "c" ): #If that key is c
                for nncar in nnCars:
                    if nncar.collided == True:
                        nnCars.remove(nncar)
                        if nncar.yaReste == False:
                            alive -= 1
            if event.key == ord ( "a" ): #If that key is a
                player = not player
            if event.key == ord ( "d" ): #If that key is d
                display_info = not display_info
            if event.key == ord ( "n" ): #If that key is n
                number_track = 2
                for nncar in nnCars:
                    nncar.velocity = 0
                    nncar.acceleration = 0
                    nncar.x = 140
                    nncar.y = 610
                    nncar.angle = 180
                    nncar.collided = False
                generateRandomMap(gameDisplay)
                bg = pygame.image.load('randomGeneratedTrackFront.png')
                bg4 = pygame.image.load('randomGeneratedTrackBack.png')

            if event.key == ord ( "b" ):
                if (len(selectedCars) == 2):
                    for nncar in nnCars:
                        nncar.score = 0
               
                    alive = num_of_nnCars
                    generation += 1
                    selected = 0
                    nnCars.clear() 
                    
                    for i in range(num_of_nnCars):
                        nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))
                        
                    for i in range(0,num_of_nnCars-2,2):
                        uniformCrossOverWeights(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])
                        uniformCrossOverBiases(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])
                    
                    nnCars[num_of_nnCars-2] = selectedCars[0]
                    nnCars[num_of_nnCars-1] = selectedCars[1]
                    
                    nnCars[num_of_nnCars-2].car_image = green_small_car
                    nnCars[num_of_nnCars-1].car_image = green_small_car
                    
                    nnCars[num_of_nnCars-2].resetPosition()
                    nnCars[num_of_nnCars-1].resetPosition()
                    
                    nnCars[num_of_nnCars-2].collided = False
                    nnCars[num_of_nnCars-1].collided = False
                                  
                    for i in range(num_of_nnCars-2):
                        for j in range(mutationRate):
                            mutateOneWeightGene(nnCars[i], auxcar)
                            mutateOneWeightGene(auxcar, nnCars[i])
                            mutateOneBiasesGene(nnCars[i], auxcar)
                            mutateOneBiasesGene(auxcar, nnCars[i])
                    if number_track != 1:
                        for nncar in nnCars:
                            nncar.x = 140
                            nncar.y = 610 
                      
                    selectedCars.clear()
                    
                
                    
                    
            if event.key == ord ( "m" ):
                if (len(selectedCars) == 2):
                    for nncar in nnCars:
                        nncar.score = 0
                   
                    alive = num_of_nnCars
                    generation += 1
                    selected = 0
                    nnCars.clear() 
                    
                    for i in range(num_of_nnCars):
                        nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))
                        
                    for i in range(0,num_of_nnCars-2,2):
                        uniformCrossOverWeights(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])
                        uniformCrossOverBiases(selectedCars[0], selectedCars[1], nnCars[i], nnCars[i+1])
                    
                    nnCars[num_of_nnCars-2] = selectedCars[0]
                    nnCars[num_of_nnCars-1] = selectedCars[1]
                    
                    nnCars[num_of_nnCars-2].car_image = green_small_car
                    nnCars[num_of_nnCars-1].car_image = green_small_car
                    
                    nnCars[num_of_nnCars-2].resetPosition()
                    nnCars[num_of_nnCars-1].resetPosition()
                    
                    nnCars[num_of_nnCars-2].collided = False
                    nnCars[num_of_nnCars-1].collided = False
                                  
                    for i in range(num_of_nnCars-2):
                        for j in range(mutationRate):
                            mutateOneWeightGene(nnCars[i], auxcar)
                            mutateOneWeightGene(auxcar, nnCars[i])
                            mutateOneBiasesGene(nnCars[i], auxcar)
                            mutateOneBiasesGene(auxcar, nnCars[i])

                    for nncar in nnCars:
                        nncar.x = 140
                        nncar.y = 610 
                      
                    selectedCars.clear()
                    
                    number_track = 2
                    for nncar in nnCars:
                        nncar.velocity = 0
                        nncar.acceleration = 0
                        nncar.x = 140
                        nncar.y = 610
                        nncar.angle = 180
                        nncar.collided = False
                    generateRandomMap(gameDisplay)
                    bg = pygame.image.load('randomGeneratedTrackFront.png')
                    bg4 = pygame.image.load('randomGeneratedTrackBack.png')
            if event.key == ord ( "r" ):
                generation = 1
                alive = num_of_nnCars
                nnCars.clear() 
                selectedCars.clear()
                for i in range(num_of_nnCars):
                    nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))
                for nncar in nnCars:
                    if number_track == 1:
                        nncar.x = 120
                        nncar.y = 480
                    elif number_track == 2:
                        nncar.x = 100
                        nncar.y = 300
            if event.key == ord ( "0" ):
                mutationRate = 0
            if event.key == ord ( "1" ):
                mutationRate = 10
            if event.key == ord ( "2" ):
                mutationRate = 20
            if event.key == ord ( "3" ):
                mutationRate = 30
            if event.key == ord ( "4" ):
                mutationRate = 40
            if event.key == ord ( "5" ):
                mutationRate = 50
            if event.key == ord ( "6" ):
                mutationRate = 60
            if event.key == ord ( "7" ):
                mutationRate = 70
            if event.key == ord ( "8" ):
                mutationRate = 80
            if event.key == ord ( "9" ):
                mutationRate = 90

        
        if event.type == pygame.MOUSEBUTTONDOWN:
            #This returns a tuple:
            #(leftclick, middleclick, rightclick)
            #Each one is a boolean integer representing button up/down.
            mouses = pygame.mouse.get_pressed()
            if mouses[0]:
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])
                #Revisar la lista de autos y ver cual estaba ahi
                for nncar in nnCars:  
                    polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
                    if (polygon.contains(point)):
                        if nncar in selectedCars:
                            selectedCars.remove(nncar)
                            selected -= 1
                            if nncar.car_image == white_big_car:
                                nncar.car_image = white_small_car 
                            if nncar.car_image == green_big_car:
                                nncar.car_image = green_small_car
                            if nncar.collided:
                                nncar.velocity = 0
                                nncar.acceleration = 0
                            nncar.update()
                        else:
                            if len(selectedCars) < 2:
                                selectedCars.append(nncar)
                                selected +=1
                                if nncar.car_image == white_small_car:
                                    nncar.car_image = white_big_car  
                                if nncar.car_image == green_small_car:
                                    nncar.car_image = green_big_car  
                                if nncar.collided:
                                    nncar.velocity = 0
                                    nncar.acceleration = 0
                                nncar.update()
                        break
            
    
            if mouses[2]:
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])
                for nncar in nnCars:  
                    polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
                    if (polygon.contains(point)):
                        if nncar not in selectedCars:
                            nnCars.remove(nncar)
                            alive -= 1
                        break

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car.rotate(-5)
    if keys[pygame.K_RIGHT]:
        car.rotate(5)
    if keys[pygame.K_UP]:
        car.set_accel(0.2)
    else:
        car.set_accel(0)
    if keys[pygame.K_DOWN]:
        car.set_accel(-0.2)
      
    redrawGameWindow()   
    
    clock.tick(FPS)




    