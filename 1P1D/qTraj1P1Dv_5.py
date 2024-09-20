import numpy as np
import os
import time
import sys
from joblib import Parallel, delayed

# FUNCTIONS
# qSystem = [nmodes, qModes, d, dd, d2, ax, rn, Qx, bb, theta, omega, openSystem, sigDp, sigDa]
# qSystemReg = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# qSystem = qSystemReg.astype(np.complex128)

def getProbDensity1d(dPhi, qSystem, x, t):
    bb = qSystem.bb
    rn = qSystem.rn
    theta = qSystem.theta
    omega = qSystem.omega
    Cn = rn * np.exp(-1*1j * theta, dtype = np.complex128)
    CnExpOmega = Cn * np.exp(-1*1j * (t * omega + dPhi), dtype = np.complex128)
    Qx = qSystem.Qx
    psi = bb * np.sum(CnExpOmega * np.sin(Qx * x))
    rho = np.abs(psi) ** 2
    return rho

def getVelocity1d(dPhi, qSystem, x, t):
    d = np.array(qSystem.d)
    rn = np.array(qSystem.rn)
    theta = np.array(qSystem.theta)
    omega = np.array(qSystem.omega)
    Cn = rn * np.exp(-1j * theta)
    Qx = np.array(qSystem.Qx)
    dPhi_64 = dPhi.astype(float)
    t_64 = t.astype(float)
    x_64 = x.astype(float)
    CnExpOmega = Cn * np.exp(-1j * (t_64 * omega + dPhi_64))
    psi = np.sum(CnExpOmega * np.sin(Qx * x_64))
    conj_psi = np.conj(psi)
    partial_Psi_x = np.sum(CnExpOmega * np.cos(Qx * x_64) * Qx)
    # conj_partial_Psi_x = np.conj(partial_Psi_x)    FIXED
    JBohmx = d * np.imag(conj_psi * partial_Psi_x)
    rho = np.abs(psi) ** 2
    vx = JBohmx / rho
    return vx

def getQuantumPE1P1D(dPhi, qSystem, x0, t):
    d = qSystem.d
    d2 = qSystem.d2
    rn = qSystem.rn
    theta = qSystem.theta
    omega = qSystem.omega
    Cn = rn * np.exp(-1j * theta, dtype = np.complex128)
    Qx = qSystem.Qx
    dx = 0.00001
    dxSQ = dx ** 2
    CnExpOmega = Cn * np.exp(-1j * (t * omega + dPhi), dtype = np.complex128)
    xN = x0 - dx
    xP = x0 + dx
    psiP = np.sum(CnExpOmega * np.sin(Qx * xP))
    psi0 = np.sum(CnExpOmega * np.sin(Qx * x0))
    psiN = np.sum(CnExpOmega * np.sin(Qx * xN))
    Rpos = np.abs(psiP)
    Rmid = np.abs(psi0)
    Rneg = np.abs(psiN)
    secondDerivativeR = (Rpos - 2 * Rmid + Rneg) / dxSQ
    quantumPE = -d2 * secondDerivativeR / Rmid
    return quantumPE

def vSearch(qSystem, xIn, vTarget):
    systemType = 0
    nmodes = qSystem.nmodes
    inTol = False
    runCount = 0
    runMax = 100000
    minDiff = np.ones(runMax) * 10 ** 100
    ampArr = np.zeros(runMax)
    phaseOut = np.ones(nmodes) * 10 ** 100

    vTolerance = 1e-12
    nSamples = 100
    rDecay = 0.99
    rGrow = 1 / rDecay ** 5

    while not inTol and runCount < runMax:
        if runCount <= 1:
            phaseAmpDeg = 100
        else:
            phaseAmpDeg = ampArr[runCount - 1]
        phaseAmp = phaseAmpDeg * np.pi / 180

        phaseShift = [None] * nSamples
        dPhi = np.zeros(nmodes)

        if systemType > 0:
            for s in range(nSamples):
                phaseShift[s] = np.zeros(nmodes)
        else:
            for s in range(nSamples):
                if runCount == 0:
                    phaseShift[s] = phaseAmp * np.random.randn(nmodes)
                else:
                    phaseShift[s] = phaseOut + phaseAmp * np.random.randn(nmodes)

        velX = np.zeros((nSamples, 1))
        to = np.array([0])
        xo = np.array([xIn])
        for s in range(nSamples):
            dPhi = phaseShift[s]
            # print(dPhi)
            # print(type(dPhi))
            # print(type(to))
            # print(type(xo))
            # print(type(qSystem))
            vxo = getVelocity1d(dPhi, qSystem, xo, to)
            # print(vxo)
            velX[s, :] = vxo

        abs_diff = np.abs(velX - vTarget)
        # print(abs_diff)
        idx = np.ravel(abs_diff)
        # print(idx)
        idx = np.argsort(idx)
        # print(idx)
        # break
        inTol = np.any(abs_diff <= vTolerance)
        minDiff[runCount] = np.min(abs_diff)
        ampArr[runCount] = phaseAmpDeg

        if runCount >= 1:
            if minDiff[runCount] > minDiff[runCount - 1]:
                ampArr[runCount] = rDecay * ampArr[runCount]
                phaseOut = phaseOutTemp
            else:
                ampArr[runCount] = rGrow * ampArr[runCount]
                phaseOut = phaseShift[idx[0]]
        else:
            ampArr[runCount] = ampArr[runCount]
            phaseOut = phaseShift[idx[0]]

        minDiff[runCount] = np.min(minDiff)
        phaseOutTemp = phaseOut
        runCount += 1

    return phaseOut, inTol

def propagateMotion1d(dPhi, qSystem, xo, vxo, tArray, h):
    dtShift = 0.01 * h
    h2 = h / 2
    h6 = h / 6
    ax = qSystem.ax
    dAbs = 0.01 * ax
    dAbs10 = dAbs / 10
    kFix = 0

    phiPts = len(dPhi)
    nPts = len(tArray)
    pX = np.zeros(nPts)
    vX = np.zeros(nPts)
    qU = np.zeros(nPts)

    pX[0] = xo
    vX[0] = vxo
    t0 = tArray[0]
    qU[0] = getQuantumPE1P1D(dPhi, qSystem, xo, t0)

    openSystem = qSystem.openSystem
    
    if openSystem:
        sigDp = qSystem.sigDp
        sigDa = 5 * qSystem.sigDa  # 5 is needed due to internal running average
        # sigDL = 5 * qSystem['sigDL']  # Uncomment if needed
        fDa = 0
        fDL = 0  # Running average fDa= 0.923*fDa + 0.077*dfDa

        for _ in range(10000):
            dfDa = sigDa * np.random.randn()
            fDa = 0.923 * fDa + 0.077 * dfDa
            # dfDL = sigDL * np.random.randn()  # Uncomment if needed
            # fDL = 0.923 * fDL + 0.077 * dfDL  # Uncomment if needed

    for nt in range(1, nPts):
        t0 = tArray[nt - 1]
        x0 = pX[nt - 1]
        vx0 = vX[nt - 1]
        tf = tArray[nt] - dtShift
        t = t0

        while t < tf:
            dt = h
            k0x = vx0
            x1 = x0 + h * k0x * (1 / 3)
            k1x = getVelocity1d(dPhi, qSystem, x1, t + h * (1 / 3))
            x2 = x0 + h * ((2 / 3) * k1x)
            k2x = getVelocity1d(dPhi, qSystem, x2, t + h * (2 / 3))
            x3 = x0 + h * ((1 / 12) * k0x + (1 / 3) * k1x + (-1 / 12) * k2x)
            k3x = getVelocity1d(dPhi, qSystem, x3, t + h * (1 / 3))
            x4 = x0 + h * ((-1 / 16) * k0x + (9 / 8) * k1x + (-3 / 16) * k2x + (-3 / 8) * k3x)
            k4x = getVelocity1d(dPhi, qSystem, x4, t + h * (1 / 2))
            x5 = x0 + h * ((9 / 8) * k1x + (-3 / 8) * k2x + (-3 / 4) * k3x + (1 / 2) * k4x)
            k5x = getVelocity1d(dPhi, qSystem, x5, t + h * (1 / 2))
            x6 = x0 + h * ((9 / 44) * k0x + (-9 / 11) * k1x + (63 / 44) * k2x + (18 / 11) * k3x + (-16 / 11) * k5x)
            k6x = getVelocity1d(dPhi, qSystem, x6, t + h)
            xx = x0 + h * ((11 / 120) * k0x + (27 / 40) * k2x + (27 / 40) * k3x + (-4 / 15) * k4x + (-4 / 15) * k5x + (
                        11 / 120) * k6x)
            dxAbs = np.abs(xx - x0)

            while xx > ax or xx < 0 or dxAbs > dAbs:
                dt = dt / 2

                if dt < 0.0001:
                    dt = 0.0001
                    xx = x0 + dAbs10 * np.random.randn(1)

                    while xx > ax or xx < 0 or dxAbs > dAbs:
                        xx = x0 + dAbs10 * np.random.randn(1)
                    #    print(xx)

                    kFix += 1

                    if kFix > 10:
                        raise ValueError('Velocity field is too erratic to integrate')
                else:
                    kFix = 0
                    hB = dt
                    h2B = dt / 2
                    h6B = hB / 6
                    x1 = x0 + k0x * h2B
                    k1x = getVelocity1d(dPhi, qSystem, x1, t + h2B)
                    x2 = x1 + k1x * h2B
                    k2x = getVelocity1d(dPhi, qSystem, x2, t + h2B)
                    x3 = x2 + k2x * hB
                    k3x = getVelocity1d(dPhi, qSystem, x3, t + hB)
                    xx = x0 + (k0x + 2 * k1x + 2 * k2x + k3x) * h6B

            t += dt
            x0 = xx

            if openSystem:
                dPhi = dPhi + qSystem.sigDp * np.random.randn(phiPts)
                dfDa = qSystem.sigDa * np.random.randn()
                fDa = 0.923 * fDa + 0.077 * dfDa
                rnNew = rn * (1 + fDa)
                probArray = rnNew ** 2
                totalProb = np.sum(probArray)
                rnNew = np.sqrt(probArray / totalProb)
                qSystem.rn = rnNew

        pX[nt] = x0
        vX[nt] = getVelocity1d(dPhi, qSystem, xx, t)
        tArray[nt] = t
        qU[nt] = getQuantumPE1P1D(dPhi, qSystem, x0, t)

    return pX, vX, qU


# Set constants
kb = 8.6183294e-5  # Boltzmann Constant in eV/K
hbar = 4.14 / (2 * np.pi)  # dimension is (eV*fs)
AngChar = 'Å'  # Angstrom character
EpsChar = 'ε'

# Conversion constants
JtoeV = 6.242e18  # J --> eV
Regc = 3e8  # speed of light in m/s
Specalc = 3e3  # speed of light in Ang/fs
AMUtoKg = 1.6605e-27  # AMU --> kg

# User input
print(' ')
# massAMU = float(input('What mass will be used? (AMU): '))
massAMU = 0.000548579909
mass = massAMU * 103.6484  # in units of eV*fs^2/Å^2
if mass < 1.0e-25:
    raise ValueError('Mass cannot be negative')

# ax = float(input('Enter box size (in Ang): '))
ax = 200
if ax < 1:
    raise ValueError('Box size must be greater than 1 Angstrom')

# oneStartNum = int(input('Enter 0,1,2,3 for {many,single,linear,epsilon} initial positions: '))
startNum = 3
if startNum == 0:
    startType = 'Prob'
elif startNum == 1:
    startType = 'Single'
elif startNum == 2:
    startType = 'Linear'
else:
    startType = f'{EpsChar} Search'

# rk4 = 0  # fix me only on rk6 as of now....
# iParity = int(input('Enter 0,1,2 for {no,odd,even} parity symmetry: '))
iParity = 0
# systemType = int(input('Enter {-1,1} to model a {closed, open} system: '))
systemType = -1
# nSamples = int(input('Enter the number of trajectories to calculate: '))
nSamples = 4
# T = float(input('Enter the temperature of the box (in Kelvin): '))
T = 350
# phaseAmpDeg = float(input('Enter initial random phase angle spread (deg): '))
phaseAmpDeg = 1000
# seed1 = int(input('Enter SEED1 to setup an initial quantum state: '))
seed1 = 111

if systemType > 0:  # => open system
    print('--------------------------------------------------------------- open systems')
    # seed2 = int(input('Enter SEED2 to simulate the diffusive process: '))
    seed2 = 321
    # Dp = float(input('Enter Dp in (rad^2)/ps for phase angle: '))
    Dp = 0.00485
    # Da = float(input('Enter Da in (percent^2)/ps on prob-amplitude: '))
    Da = 0.0
    # DL = float(input('Enter DL in (percent^2)/ps on box side Length: '))
    DL = 0
else:
    Dp = 0
    Da = 0
    DL = 0

# Enforce parity symmetry
if iParity == 0:
    nx0 = 1
    dnx = 1
elif iParity == 1:
    nx0 = 1
    dnx = 2
elif iParity == 2:
    nx0 = 2
    dnx = 2
else:
    raise ValueError('Parity type is not specified!')
# Set type of quantum system
openSystem = systemType > 0

# Set type of integrator (RK4 vs RK6)
# rk4 = True if rk4 > 0 else False

# Start setup
np.random.seed(seed1)
d = hbar / mass
dd = (hbar ** 2 * np.pi ** 2) / (2 * mass)
d1 = (hbar * np.pi ** 2) / (2 * mass)  # d1 = dd/hbar = (hbar*pi^2)/(2*mass)
d2 = (hbar ** 2) / (2 * mass)
vRMS1d = np.sqrt(kb * T / mass)  # in 1D: classical RMS velocity
vmax = 10 * vRMS1d

# Estimate maximum quantum number
imax = round((mass * vmax * ax) / (np.pi * hbar))
imax = max(imax, 3)
xShift = 0.05 * ax
xLow = 0 - xShift
xBig = ax + xShift

# Sanity checks
if nSamples < 1 or nSamples > 1000:
    print('nSamples must be at least 1 and not greater than 1000')
if vmax > 300:
    print('Lower the temperature: Relativistic effects are appearing.')

# Get Cnm and Qx
kT = kb * T
qModes = np.arange(nx0, imax + 1, dnx)  # list of quantum modes (qModes)
Estates = dd * (qModes / ax) ** 2
Eground = dd * (nx0 / ax) ** 2
dEstates = Estates - Eground
sWeights = np.exp(-dEstates / kT)  # statistical weights

totalWeight = np.sum(sWeights)
prob = sWeights / totalWeight
Cprob = np.cumsum(prob)
L = Cprob < 0.9999995
nmodes = max(np.sum(L), 3)  # must track at least 3 modes
qModes = qModes[:nmodes]
Estates = dd * (qModes / ax) ** 2
omega = Estates / hbar
rn = np.sqrt(prob[:nmodes])  # note: rn = abs(Cn)
theta = 2 * np.pi * np.random.rand(nmodes)  # fix the initial random phase
# Cn = rn * np.exp(1j * theta)  # apply phase
Qx = (np.pi * qModes) / ax

# Sanity check
KEclassical = 0.5 * mass * vRMS1d ** 2
r2total = np.sum(rn ** 2)
if abs(r2total - 1) > 0.000001:
    # print('Cn coefficients were renormalized!')
    rn = rn / np.sqrt(r2total)
    r2total = np.sum(rn ** 2)
    if abs(r2total - 1) > 0.000001:
        print('Cn coefficients cannot be normalized!')

aveE0 = d2 * np.sum((rn * Qx) ** 2)
percentError = 100 * (KEclassical - aveE0) / aveE0

# Prevent negative diffusion coefficients
if Dp < 1.0e-35:  # for phase
    Dp = 0
if Da < 1.0e-35:  # for amplitude
    Da = 0
# if DL < 1.0e-35:  # for side of box
#     DL = 0

# Give warning
if nmodes > 5000:
    print(f'nmodes = {nmodes}')
    print('Warning: Number of modes is > 5000. compute time will be long.')

Emax = np.max(Estates)
# Scale the time of simulation
dEmove = aveE0 - Eground
vx = np.sqrt(2 * dEmove / mass) + 1.0e-10
Tx = min(ax / vx, 10000000)  # time scale for particle to cross side of box
mObs = 1000
dtObsv = Tx / mObs  # this program sets this very small as a check
# Nsweeps = int(input('Enter the number of sweeps to be observed: '))
Nsweeps = 10
print(f'  dt for observations = {dtObsv} fs ')
nObservations = Nsweeps * mObs
# dtRK4 = float(input('Enter the dt (fs) for solving dif-eq using RK4: '))
dtRK4 = dtObsv / 4
runTime = Nsweeps * Tx
nRK4 = np.ceil(dtObsv / dtRK4)
dtRK4 = dtObsv / nRK4

# Define diffusion coefficients for each process for an open system
if systemType > 0:
    # Set diffusion rates
    sigDp = np.sqrt(dtRK4 * Dp / 1000)  # => sqrt((dtRK4*sig0^2)/1000)
    sigDa = np.sqrt(dtRK4 * Da / 1000)
    # sigDL = np.sqrt(dtRK4 * DL / 1000)
else:
    sigDp = 0
    sigDa = 0
    # sigDL = 0

# Create output log and data file names
strMass = str(massAMU).replace('.', '_')
strDtRK4 = str(dtRK4).replace('.', '_')

if openSystem:
    strDp = str(Dp).replace('.', '_')
    strDa = str(Da).replace('.', '_')
    strPs = str(phaseAmpDeg).replace('.', '_')
    baseName = f'1P1D{T}Lx{ax}s{seed1}RK4dt{strDtRK4}m{strMass}Ps{strPs}s{seed2}Dp{strDp}Da{strDa}open'
else:
    baseName = f'1P1D{T}Lx{ax}s{seed1}RK4dt{strDtRK4}m{strMass}Ps{phaseAmpDeg}closed'

# Minimum check on output folder and file names
folderName = '1P1D'
currentFolder = os.getcwd()
testEnd = currentFolder[-4:]

if testEnd == folderName:  # already in the data folder
    baseFileName = baseName  # no pre-appending folder name is necessary
else:  # in directory above data folder
    # Create directory if it doesn't exist
    if not os.path.isdir(folderName):
        os.mkdir(folderName)
    
    folderPrefix = '1P1D/' if os.name == 'posix' else '1P1D\\'
    baseFileName = f'{folderPrefix}{baseName}'  # add folder path to name

fileNameP_x = f'{baseFileName}P_x.txt'
fileNameV_x = f'{baseFileName}V_x.txt'
fileNameQpe = f'{baseFileName}Qpe.txt'
flagError = -1

if os.path.isfile(fileNameP_x):
    flagError = 1
if os.path.isfile(fileNameV_x):
    flagError = 1
if os.path.isfile(fileNameQpe):
    flagError = 1

# if flagError > 0:
#     raise RuntimeError('This simulation has been done previously. No auto-overwrite')

bb = np.sqrt(2/ax)

# Build system quantum wavefunction
class QuantumSystem:
    def __init__(self, nmodes, qModes, d, dd, d2, ax, rn, Qx, bb, theta, omega, openSystem, sigDp, sigDa):
        self.nmodes = nmodes
        self.qModes = qModes
        self.d = d
        self.dd = dd
        self.d2 = d2
        self.ax = ax
        self.rn = rn  # note: rn = abs(Cn) where Cn are expansion coefficients
        self.Qx = Qx
        self.bb = bb
        self.theta = theta  # initial phase on expansion coefficients
        self.omega = omega  # frequencies for each mode
        self.openSystem = openSystem  # true/false toggle for open system
        self.sigDp = sigDp  # STD for accumulative Gaussian process
        self.sigDa = sigDa  # STD for stationary Gaussian process
        # self.sigDL = sigDL  # STD for stationary Gaussian process
        # self.rk4 = rk4
    def _ensure_complex(self, value):
        if isinstance(value, (list, np.ndarray)):
            return np.array(value, dtype=np.complex128)
        else:
            return complex(value)
    def __str__(self):
        return f"nmodes: {self.nmodes}, qModes: {self.qModes}, d: {self.d}, dd: {self.dd}, d2: {self.d2}, " \
               f"ax: {self.ax}, rn: {self.rn}, Qx: {self.Qx}, bb: {self.bb}, theta: {self.theta}, " \
               f"omega: {self.omega}, openSystem: {self.openSystem}, sigDp: {self.sigDp}, sigDa: {self.sigDa}"

qSystem = QuantumSystem(nmodes, qModes, d, dd, d2, ax, rn, Qx, bb, theta, omega, openSystem, sigDp, sigDa)

# print(qSystem)

# Generate initial random phases for each sample
phaseAmp = phaseAmpDeg * np.pi / 180  # convert degrees to radians
phaseShift = [np.zeros(nmodes) for _ in range(nSamples)]
dPhi = np.zeros(nmodes)  # simply to preallocate
# print(type(phaseShift))
# print(type(dPhi))

if systemType > 0:  # => open system
    for s in range(nSamples):
        phaseShift[s] = np.zeros(nmodes)
else:  # systemType < 0 => closed system
    for s in range(nSamples):
        phaseShift[s] = phaseAmp * np.random.randn(nmodes)  # units in radians

# For linear start ensure all phaseShift array is equal
# Can comment this out to redo independent particle/ideal gas
dPhi1 = phaseShift[0]

if startNum == 2:
    # Replicate the data across all other cells
    for i in range(1, len(phaseShift)):
        phaseShift[i] = dPhi1


# Initialize variables
nGrid = 50000
dx = ax / nGrid
x0 = dx / 2
xArray = np.arange(x0, ax + dx, dx)
iPlaces = len(xArray)
rhoArray = np.zeros(iPlaces)
xLocate = np.zeros(iPlaces)
aveVx = 0
aveVx2 = 0
aveQPE = 0
tTryArray = np.arange(0, 101, 10)
reducedProb = 0

# Calculate equilibrium properties
for tTry in tTryArray:
    rhoArray = np.array([getProbDensity1d(dPhi1, qSystem, x, tTry) for x in xArray])
    xLocate = xArray

    # Calculate average properties
    prob = rhoArray * dx
    totalProb = np.sum(prob)  # should be 1 if grid resolution is high enough
    prob = prob / totalProb

    # Extract initial starting positions
    if tTry == 0:
        save_xLocate = xLocate
        save_prob = prob

    # Sample places
    nOutliers = 0
    for i, x in enumerate(xLocate):
        kx = getVelocity1d(dPhi, qSystem, x, tTry)
        quantumPE = getQuantumPE1P1D(dPhi, qSystem, x, tTry)
        if abs(quantumPE) < 100 * Emax and kx < 2000:  # removes outliers
            aveVx += prob[i] * kx
            aveVx2 += prob[i] * kx * kx
            aveQPE += prob[i] * quantumPE
            reducedProb += prob[i]
        else:
            nOutliers += 1

aveVx2 = aveVx2 / reducedProb
meanKE = 0.5 * mass * aveVx2
aveQPE = aveQPE / reducedProb
aveBohmEnergy = meanKE + aveQPE  # note avePE = 0 classical contribution
percentError2 = 100 * (aveBohmEnergy - aveE0) / aveE0

# Set up time of observations
tArray = np.arange(0, runTime + dtObsv, dtObsv)
runTime = tArray[-1]
nPts = len(tArray)

# Place test particles in the system
if startNum == 0:  # let us randomize particle location too
    indx = np.argsort(save_prob)[::-1]
    mTemp = np.ceil(iPlaces / 2)
    dmS = max(np.floor(mTemp / nSamples).astype(int), 1)
    mSmax = nSamples * dmS
    xoArray = save_xLocate[indx[:mSmax:dmS]]
elif startNum == 1:
    xo = ax / np.sqrt(2)
    xoArray = np.full(nSamples, xo)  # only one location
elif startNum == 2:
    linDx = 1 / nSamples
    xoArray = np.linspace(linDx / 2, ax - linDx / 2, nSamples)
elif startNum == 3:
    xMew = ax / np.sqrt(2)
    vMew = 0.1
    epsilon_x = 1e-10
    epsilon_v = 1e-10
    xoArray = xMew + epsilon_x * np.random.randn(nSamples)
    vDist = vMew + epsilon_v * np.random.randn(nSamples)
    stdPhi = np.zeros(nSamples)
    for s in range(nSamples):
        phaseOut, inTol = vSearch(qSystem, xoArray[s], vDist[s])
        if inTol == 1:
            phaseShift[s] = phaseOut
            stdPhi[s] = np.std(phaseOut)
        else:
            raise ValueError("Could not find a solution!")
        
# Set open system seed2
if systemType > 0:
    np.random.seed(seed2)  # reset random numbers for diffusive process
else: 
    seed2=0


# Function to display parameters
def summarize_parameters(systemType, baseName, seed1, mass, massAMU, T, vRMS1d, ax, Tx, Nsweeps, runTime, dtObsv, dtRK4, nRK4, nObservations, startNum, xo, nx0, dnx, nmodes, Emax, nSamples, seed2, phaseAmpDeg, Dp, Da, KEclassical, aveE0, meanKE, aveQPE, aveBohmEnergy, nOutliers, percentError, percentError2):
    if systemType > 0:
        systemTypeName = 'open system'
    else:
        systemTypeName = 'closed system'
    
    print('   ')
    print(f'             base file name= {baseName}')
    print(f'  initial conditions: SEED1= {seed1}')
    print(f'                       mass= {mass} evfs^2/Ang^2')
    print(f'                       mass= {massAMU:.9f} AMU')
    print(f'                          T= {T} Kelvin')
    print(f'           classical vRMS1d= {vRMS1d} Ang/fs')
    print(f'                         ax= {ax} Ang')
    print(f'               sweep period= {Tx} fs')
    print(f'                    Nsweeps= {Nsweeps}')
    print(f'                       Trun= {runTime:.9f} fs')
    print(f'                     dtObsv= {dtObsv:.9f} fs')
    print(f'                      dtRK4= {dtRK4:.9f} fs')
    print(f'        dtObsv subdivisions= {nRK4}  => # of loops per observation')
    print(f'    total # of observations= {nObservations}')
    
    if startNum == 1:
        print(f'the single initial position= {xo} Ang')
    elif startNum == 0:
        print(' use many initial positions: randomize over |psi|^2')
    elif startNum == 2:
        print('set of ICs as a linear spacing across the box')
    else:
        print('set of ICs as a controlled epsilon bubble')
    
    print(f'  parity symmetry type: nx0= {nx0}  dnx= {dnx}')
    print(f'        number of Cnm terms= {nmodes}')
    print(f'maximum energy state: Emax = {Emax} eV')
    print(f'                   nSamples= {nSamples}')
    print('----------------------------------------------------------------')
    print(f'            simulation type= {systemTypeName}')
    
    if systemType > 0:
        print(f' diffusive processes: SEED2= {seed2}')
        print(f' initial random phase stdev= {phaseAmpDeg}  deg')
        print(f'  phase angle diffusion: Dp= {Dp:.9f}  (degree^2)/ps')
        print(f'   amplitude modulation: Da= {Da:.9f}  (percnt^2)/ps')
    
    print(f'                KEclassical= {KEclassical:.9f} eV')
    print(f'     Copenhagen mean energy= {aveE0:.9f} eV')
    print(f'                mean BohmKE= {meanKE:.9f} eV')
    print(f'             mean quantumPE= {aveQPE:.9f} eV')
    print(f'              aveBohmEnergy= {aveBohmEnergy:.9f} eV')
    print(f'  number of energy outliers= {nOutliers}')
    print(f'Copenhagen/classical %error= {percentError}')
    print(f'  Copenhagen/Bohmian %error= {percentError2}')

# Example call to the function

if startNum == 1:
    summarize_parameters(
        systemType, baseName, seed1, mass, massAMU, T, vRMS1d, ax, Tx, Nsweeps, runTime, 
        dtObsv, dtRK4, nRK4, nObservations, startNum, xo, nx0, dnx, nmodes, Emax, nSamples, 
        seed2, phaseAmpDeg, Dp, Da, KEclassical, aveE0, meanKE, aveQPE, aveBohmEnergy, nOutliers, 
        percentError, percentError2
    )
else:
    xo=0 
    summarize_parameters(
        systemType, baseName, seed1, mass, massAMU, T, vRMS1d, ax, Tx, Nsweeps, runTime, 
        dtObsv, dtRK4, nRK4, nObservations, startNum, xo, nx0, dnx, nmodes, Emax, nSamples, 
        seed2, phaseAmpDeg, Dp, Da, KEclassical, aveE0, meanKE, aveQPE, aveBohmEnergy, nOutliers, 
        percentError, percentError2
    )


# Initialize variables
tArray = np.arange(0, runTime + dtObsv, dtObsv)
nPts = len(tArray)
posX = np.zeros((nSamples, nPts))
velX = np.zeros((nSamples, nPts))
q_PE = np.zeros((nSamples, nPts))
# print(posX)
pX = np.zeros((1, nPts))
vX = np.zeros((1, nPts))
qU = np.zeros((1, nPts))

# Timing the computation
start_time = time.time()

# Parallel computation for each sample
def compute_sample(s):
    # xo = xoArray[s]
    # print(xoArray)
    # dPhi = phaseShift[s]
    # print(dPhi)
    vxo = getVelocity1d(phaseShift[s], qSystem, xoArray[s], tArray[0])
    pX, vX, qU = propagateMotion1d(phaseShift[s], qSystem, xoArray[s], vxo, tArray, dtRK4)
    # print('zzz')
    return pX, vX, qU

# print (nSamples)

# results = Parallel(n_jobs=-1)(delayed(compute_sample)(s) for s in range(nSamples))
# results = ((compute_sample)(s) for s in range(nSamples))
for s in range(0,nSamples):
    result = compute_sample(s)
    # print(result)
    posX[s, :] = result[0]
    velX[s, :] = result[1]
    q_PE[s, :] = result[2]
    # print('hi')

# for s, (pX, vX, qU) in enumerate(results):
#     posX[s, :] = pX
#     velX[s, :] = vX
#     q_PE[s, :] = qU
#     print('hi')

# print(posX)

cpu_time = time.time() - start_time
print(f'                CPU seconds= {cpu_time:.2f}')

# Write results to output data files
np.savetxt(f'{fileNameP_x}', posX, delimiter=',')
np.savetxt(f'{fileNameV_x}', velX, delimiter=',')
np.savetxt(f'{fileNameQpe}', q_PE, delimiter=',')

# Write results to output log file
log_file_name = f'{baseName}Log.txt'
with open(log_file_name, 'w') as fid:
    fid.write(f'{baseName}\n')
    fid.write('simulation of quantum trajectories for confined particles\n')
    fid.write('-------------------------------------------------------------- summary\n')
    fid.write(f'  initial conditions: SEED1= {seed1}\n')
    fid.write(f'                       mass= {mass} evfs^2/Ang^2\n')
    fid.write(f'                       mass= {massAMU:.9f} AMU\n')
    fid.write(f'                          T= {T} Kelvin\n')
    fid.write(f'           classical vRMS1d= {vRMS1d} Ang/fs\n')
    fid.write(f'                         ax= {ax} Ang\n')
    fid.write(f'               sweep period= {Tx} fs\n')
    fid.write(f'                    Nsweeps= {Nsweeps}\n')
    fid.write(f'                       Trun= {runTime:.9f} fs\n')
    fid.write(f'                     dtObsv= {dtObsv:.9f} fs\n')
    fid.write(f'                      dtRK4= {dtRK4:.9f} fs\n')
    fid.write(f'        dtObsv subdivisions= {nRK4}  => # of loops per observation\n')
    fid.write(f'    total # of observations= {nObservations}\n')

    if startNum == 1:
        fid.write(f'the single initial position= {xo:.9f} Ang\n')
    elif startNum == 0:
        fid.write(' use many initial positions: randomize over |psi|^2\n')
    elif startNum == 2:
        fid.write('set of ICs as a linear spacing across the box\n')
    else:
        fid.write('set of ICs as a controlled epsilon bubble\n')

    fid.write(f'  parity symmetry type: nx0= {nx0}  dnx= {dnx}\n')
    fid.write(f'        number of Cnm terms= {nmodes}\n')
    fid.write(f'maximum energy state: Emax = {Emax} eV\n')
    fid.write(f'                   nSamples= {nSamples}\n')
    fid.write('----------------------------------------------------------------\n')
    fid.write(f'            simulation type= {"open system" if systemType > 0 else "closed system"}\n')
    if systemType > 0:
        fid.write(f' diffusive processes: SEED2= {seed2}\n')
        fid.write(f' initial random phase stdev= {phaseAmpDeg}  deg\n')
        fid.write(f'  phase angle diffusion: Dp= {Dp:.9f}  (degree^2)/ps\n')
        fid.write(f'   amplitude modulation: Da= {Da:.9f}  (percnt^2)/ps\n')
    fid.write(f'                KEclassical= {KEclassical:.9f} eV\n')
    fid.write(f'     Copenhagen mean energy= {aveE0:.9f} eV\n')
    fid.write(f'                mean BohmKE= {meanKE:.9f} eV\n')
    fid.write(f'             mean quantumPE= {aveQPE:.9f} eV\n')
    fid.write(f'              aveBohmEnergy= {aveBohmEnergy:.9f} eV\n')
    fid.write(f'  number of energy outliers= {nOutliers}\n')
    fid.write(f'Copenhagen/classical %error= {percentError}\n')
    fid.write(f'  Copenhagen/Bohmian %error= {percentError2}\n')


