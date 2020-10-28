import numpy as np
import itertools

'''
This script generates MD++ supercells for core energy calculations
The dislocation plane normal is along the x-direction
The dislocation line is along the z-direction
Nicolas Bertin, 08/31/2020
Yichen Qian, revised from the dislocation plane normal at x-direction to y-direction
'''

def find_burgers_coord(theta, c1, n1, c3, n3):
    
    # compute Burgers vector in scaled coordinates
    d1 = n1*c1
    d3 = n3*c3
    
    det = d1[0]*d3[1]-d1[1]*d3[0]
    if np.abs(det) > 1e20:
        a = 0.5*(d3[1]-d3[0])/det
        b = 0.5*(d1[0]-d1[1])/det
    else:
        det = d1[2]*d3[1]-d1[1]*d3[2]
        a = 0.5*(d3[1]-d3[2])/det
        b = 0.5*(d1[2]-d1[1])/det
        
    # make sure always remove atom, not insert atom
    if theta < 0:
        a = -a
        b = -b
    
    return np.array([0.,a,b])


def generate_script(celldata, theta):
    
    # Find corresponding cell
    ind = np.argwhere(np.abs(celldata['theta']-theta)<1e-2)
    if ind.size == 0:
        raise Exception('Cannot find character angle in the cell data')
    ind = ind[0]
        
    angle = celldata['theta'][ind]
    c1 = celldata['c1'][ind][0]
    n1 = celldata['n1'][ind]
    c2 = celldata['c2'][ind][0]
    n2 = celldata['n2'][ind]
    c3 = celldata['c3'][ind][0]
    n3 = celldata['n3'][ind]
    bs = celldata['bs'][ind][0]
    
    # Generate MD++ script
    script =  '# -*-shell-script-*-\n'
    script += '#MD++ script to compute core energies\n'
    script += 'setnolog\n'
    script += 'setoverwrite\n'
    script += 'dirname = runs/KIM/dipole_%.2f_ref\n' % angle
    script += '#------------------------------------------------------------\n'
    script += '#Read in EAM/MEAM potential\n'
    script += '#potfile = "~/Potentials/w_version3.eam" eamgrid = 5000 readeam\n'
    script += 'potfile = ~/Documents/Codes/MD++/potentials/EAMDATA/eamdata.W.Marinica13 eamgrid = 80001 readeam\n'
    script += 'NNM = 100\n'
    script += '#------------------------------------------------------------\n'
    script += 'latticestructure = body-centered-cubic\n'
    script += 'latticeconst = 3.14339 # (A) for W_cea\n'
    script += '\n'
    script += 'makecnspec = [%4d %4d %4d %4d   #(x) dipole direction\n' % (c1[0], c1[1], c1[2], n1)
    script += '              %4d %4d %4d %4d   #(y)\n' % (c2[0], c2[1], c2[2], n2)
    script += '              %4d %4d %4d %4d ] #(z) dislocation line\n' % (c3[0], c3[1], c3[2], n3)
    script += '\n'
    script += 'makecn finalcnfile = perf.cn writecn\n'
    script += '#-------------------------------------------------------------\n'
    script += '#Create Dislocation Dipole by using linear elastic solutions\n'
    script += '\n'
    script += 'mkdipole = [ 3 1 #z(dislocation line), y(dipole direction)\n'
    script += '             %12.8f %12.8f %12.8f  #(bx,by,bz)\n' % (bs[0], bs[1], bs[2])
    script += '             -0.01 -0.2499 0.251  #(x0,y0,y1) #type (2)\n'
    script += '             0.278 -10 10 -10 10  1 ] #nu, number of images, shiftbox\n'
    script += '\n'
    script += 'makedipole finalcnfile = makedp_%.2f.lammps writeLAMMPS\n' % angle
    script += '#-------------------------------------------------------------\n'
    script += '#Conjugate-Gradient relaxation\n'
    script += 'conj_ftol = 1e-7  conj_fevalmax = 3000\n'
    script += 'conj_fixbox = 1 conj_dfpred = 1e-4\n'
    script += 'relax\n'
    script += 'eval\n'
    script += 'finalcnfile = dipole_%.2f.lammps writeLAMMPS\n' % angle
    script += 'quit\n'
    
    return script



# supercell size
ar = 1.5 # aspect ratio x/y
n1 = 10.0 # supercell size along the y-direction
n3 = 3.0 # supercell size along the z-direction
mult = 3.0 # multiplication factor

bv = np.array([1,1,1]) # Burgers vector direction
c2 = np.array([-1,1,0]) # dislocation plane index

# maximum Miller index of repeat vectors allowed to
# generate supercells of various character angles
nmax = 10



# generate in-plane discrete directions
p = np.array(list(itertools.permutations(range(1, nmax+1), 2)))
p = np.vstack(([1,0], [1,1], [0,1], p))
m = np.gcd(p[:,0], p[:,1])
p = p / m[:, np.newaxis]

# generate global supercell repeat vectors
if np.abs(np.dot(bv, c2)) > 1e-5:
    raise Exception('Burgers vector and dislocation plane must be orthogonal')

y0 = np.cross(c2, bv)
my = np.gcd(y0[0], np.gcd(y0[1], y0[2]))
y0 = y0 / my
x = bv / np.linalg.norm(bv)
y = y0 / np.linalg.norm(y0)
c3plus = np.outer(p[:,0], bv) + np.outer(p[:,1], y0)
c3minus = np.outer(p[:,0], bv) - np.outer(p[:,1], y0)
c3 = np.unique(np.vstack((c3plus, c3minus)), axis=0)

# compute character angles
c3n = np.linalg.norm(c3, axis=1)
c3x = np.dot(c3, x)
c3y = np.dot(c3, y)
angle = np.arctan2(c3y, c3x)*180.0/np.pi
ia = np.argsort(angle)
angle = angle[ia]
c3 = c3[ia]

# compute complementary supercell repeat vector
c1 = np.cross(c3, c2)

# determine supercell size
m1 = np.gcd(c1[:,0], np.gcd(c1[:,1], c1[:,2]))
cm1 = c1 / m1[:, np.newaxis]
l1 = np.linalg.norm(cm1, axis=1)
n1 = np.ceil(mult*n1/l1)

m3 = np.gcd(c3[:,0], np.gcd(c3[:,1], c3[:,2]))
cm3 = c3 / m3[:, np.newaxis]
l3 = np.linalg.norm(cm3, axis=1)
n3 = np.ceil(mult*n3/l3)

# adjust aspect ratio
cm2 = np.tile(c2, (c3.shape[0], 1))
l2 = np.linalg.norm(cm2, axis=1)
n2 = np.round(ar*n1*l1/l2)

# select orientations with acceptable Miller indices
cmax = np.max(np.abs(np.hstack((cm1,cm2,cm3))), axis=1)
ind = (cmax<=nmax)


# Burgers vector in scaled coordinates
bs = np.zeros((angle.size,3))
for i in range(angle.size):
    bs[i] = find_burgers_coord(angle[i], cm1[i], n1[i], cm3[i], n3[i])


# all supercells data
celldata = {
    "theta": angle[ind],
    "c1": cm1[ind],
    "n1": n1[ind],
    "c2": cm2[ind],
    "n2": n2[ind],
    "c3": cm3[ind],
    "n3": n3[ind],
    "bs": bs[ind]
}


# Generate MD++ script for a given character angle
theta = 90.0 # edge dislocation
#theta = 0.0 # screw dislocation
#theta = 70.53 # M111 dislocation

script = generate_script(celldata, theta)
print(script)

if 0:
    # Print MD++ script into file
    script_file = open('/Users/bertin1/Documents/Codes/MD++/scripts/KIM/W-dipole_test.script', 'w')
    script_file.write(script)
    script_file.close()
