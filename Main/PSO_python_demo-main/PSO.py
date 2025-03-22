import numpy as np
from scipy.fftpack import fft 
# Why this 'fft'? 
# See: https://iphysresearch.github.io/blog/post/signal_processing/fft/#efficiency-of-the-algorithms

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.io import savemat
# from numba import jit
# from numba import njit, prange
from multiprocessing import Pool
# 常量定义
G = 6.67430e-11  # 万有引力常数, m^3 kg^-1 s^-2
c = 2.998e8  # 光速, m/s
M_sun = 1.989e30  # 太阳质量, kg
pc = 3.086e16  # pc到m的转换

__all__ = ['crcbqcpsopsd', 
           'crcbpso', 
           'crcbgenqcsig', 
           'glrtqcsig4pso',
           'ssrqc',
           'normsig4psd',
           'innerprodpsd',
           's2rv',
           'crcbchkstdsrchrng']


def crcbqcpsopsd(inParams, psoParams, nRuns):
    nSamples = len(inParams['dataX'])
    nDim = 6
    fHandle = lambda x, returnxVec: glrtqcsig4pso(x, inParams, returnxVec)

    outStruct = [{} for _ in range(nRuns)]
    outResults = {
        'allRunsOutput': [],
        'bestRun': None,
        'bestFitness': None,
        'bestSig': np.zeros(nSamples),
        # 'bestQcCoefs': np.zeros(3),
        # 'bestSNR': None,
        # 'bestAmp':None,
        'r': None,
        'm_c': None,
        'tc': None,
        'phi_c': None,
        'mlz': None,
        'y': None,
    }
    # print(f"r:{r}, m_c:{m_c}, tc:{tc}, phi_c:{phi_c}, w:{w}, y:{y}")
    # Allocate storage for outputs: results from all runs are stored
    outStruct = [outStruct for _ in range(nRuns)]

    # Independent runs of PSO [TODO: runing in parallel.]
    for lpruns in range(nRuns):
        # Reset random number generator for each run
        np.random.seed(lpruns)
        outStruct[lpruns] = crcbpso(fHandle, nDim, **psoParams)
        # Below codes are used for checking qcCoefs for current lprun.
        # _, qcCoefs = fHandle(outStruct[lpruns]['bestLocation'][np.newaxis,...], returnxVec=1)
        # print(qcCoefs)

    # Prepare output
    fitVal = np.zeros(nRuns)
    for lpruns in range(nRuns):
        allRunsOutput = {
            'fitVal': 0,
            # 'qcCoefs': np.zeros(3),
            'r': 0,
            'm_c': 0,
            'tc': 0,
            'phi_c': 0,
            'mlz': 0,
            'y': 0,
            'estSig': 0,
            'totalFuncEvals': [],
            # 'snr': [],
        }
        fitVal[lpruns] = outStruct[lpruns]['bestFitness']
        allRunsOutput['fitVal'] = fitVal[lpruns]
        _, params = fHandle(outStruct[lpruns]['bestLocation'][np.newaxis, ...], returnxVec=1)
        r = params[0, 0]
        m_c = params[0, 1]
        tc = params[0, 2]
        phi_c = params[0, 3]
        mlz = params[0, 4]
        y = params[0, 5]
        # print(f"r:{r}, m_c:{m_c}, tc:{tc}, phi_c:{phi_c}, mlz:{mlz}, y:{y}")
        # allRunsOutput['qcCoefs'] = qcCoefs[0]
        allRunsOutput['r'] = r
        allRunsOutput['m_c'] = m_c
        allRunsOutput['tc'] = tc
        allRunsOutput['phi_c'] = phi_c
        allRunsOutput['mlz'] = mlz
        allRunsOutput['y'] = y
        estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, mlz, y)  # changed by ywq
        # estSig = crcbgenqcsig(inParams['dataX'], 1, qcCoefs[0])
        estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdPosFreq'], 1)  # changed by ywq
        # estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdPosFreq'], 1)
        estAmp = innerprodpsd(inParams['dataY'], estSig, inParams['sampFreq'],
                              inParams['psdPosFreq'])  # question? 加权内积不能获得振幅吧
        estSig = estAmp * estSig
        allRunsOutput['estSig'] = estSig
        allRunsOutput['totalFuncEvals'] = outStruct[lpruns]['totalFuncEvals']
        outResults['allRunsOutput'].append(allRunsOutput)

    # 找到最佳运行
    bestRun = np.argmin(fitVal)
    outResults['bestRun'] = bestRun
    outResults['bestFitness'] = outResults['allRunsOutput'][bestRun]['fitVal']
    outResults['bestSig'] = outResults['allRunsOutput'][bestRun]['estSig']

    # 存储最佳运行的所有参数
    outResults['r'] = outResults['allRunsOutput'][bestRun]['r']
    outResults['m_c'] = outResults['allRunsOutput'][bestRun]['m_c']
    outResults['tc'] = outResults['allRunsOutput'][bestRun]['tc']
    outResults['phi_c'] = outResults['allRunsOutput'][bestRun]['phi_c']
    outResults['mlz'] = outResults['allRunsOutput'][bestRun]['mlz']
    outResults['y'] = outResults['allRunsOutput'][bestRun]['y']
    return outResults, outStruct

def crcbpso(fitfuncHandle, nDim, O=0, **varargin):
    psoParams = dict(
    popsize = 40,
    # popSize=40,  # Just in case  粒子数量
    maxSteps=2000,  # 最大迭代次数
    c1=2,
    c2=2,  # 两个学习因子
    max_velocity=0.5,  # 最大速度
    # maxVelocity=0.5,  # Just in case
    dcLaw_a=0.9,  # 动态惯性权重
    # startInertia=0.9,  # Just in case 权重初始值
    dcLaw_b=0.4,  # 动态惯性权重
    # endInertia=0.4,  # Just in case 权重结束值？
    dcLaw_c=2000 - 1,  # maxSteps-1 动态惯性权重的调整参数，用于计算惯性权重的变化
    dcLaw_d=0.2,
    bndryCond='',
    # boundaryCond='',  # Just in case 边界条件
    nbrhdSz=3,  # 邻域大小(试了一下，改变这个为29会使运行速度降低，结果上误差反而没什么变化)
)

    # Default for the level of information returned in the output
    outputLvl = 0  # 输出等级？
    returnData = {
        'totalFuncEvals': [],  # 总迭代次数
        'bestLocation': np.zeros((1, nDim)),  # 最优位置
        'bestFitness': [],  # 最优适应度
    }

    # Override defaults if needed
    nreqArgs = 2  # Not used (He Wang)
    # updata()是python字典中的内置方法，用于更新字典，即将另一个字典中可迭代的对象中的元素添加到当前字典中，重复键会自动覆盖
    psoParams.update(varargin)
    # pop()是python字典中的内置方法，用于删除字典中指定键的键值对，并返回该键对应的值，如果键不存在，则返回默认值
    # 确定变量，将变量进行统一
    # psoParams['popsize'] = psoParams.pop('popSize')
    # psoParams['max_velocity'] = psoParams.pop('maxVelocity')
    # psoParams['dcLaw_a'] = psoParams.pop('startInertia')
    # psoParams['dcLaw_b'] = psoParams.pop('endInertia')
    # psoParams['bndryCond'] = psoParams.pop('boundaryCond')
    # Neither maxInitialVelocity nor max_initial_velocity is used here.
    
    if O==1:
        returnData['allBestFit'] = np.zeros(psoParams['maxSteps'])
    elif O==2:
        returnData['allBestLoc'] = np.zeros((psoParams['maxSteps'],nDim))
    #Add more fields with additional case
    assert O<=2, 'Output level > 2 not implemented'    
    
    
    #Number of left and right neighbors. Even neighborhood size is split
    #asymmetrically: More right side neighbors than left side ones.
    psoParams['nbrhdSz'] = max([psoParams['nbrhdSz'],3])
    leftNbrs = np.floor((psoParams['nbrhdSz']-1)/2)  # Not used (He Wang)
    rightNbrs = psoParams['nbrhdSz']-1-leftNbrs      # Not used (He Wang)

    #Information about each particle stored as a row of a matrix ('pop').
    #Specify which column stores what information.
    #(The fitness function for matched filtering is SNR, hence the use of 'snr'
    #below.)
    partCoordCols = np.arange(nDim) # Particle location
    partVelCols = np.arange(nDim,2*nDim) # Particle velocity
    partPbestCols = np.arange(2*nDim,3*nDim) # Particle pbest
    partFitPbestCols = 3*nDim # Fitness value at pbest
    partFitCurrCols = partFitPbestCols+1 # Fitness value at current iteration
    partFitLbestCols = partFitCurrCols+1 # Fitness value at local best location
    partInertiaCols = partFitLbestCols+1 # Inertia weight
    partLocalBestCols = np.arange(partInertiaCols,partInertiaCols+nDim) # Particles local best location
    partFlagFitEvalCols = partLocalBestCols[-1]+1 # Flag whether fitness should be computed or not
    partFitEvalsCols = partFlagFitEvalCols+1 # Number of fitness evaluations

    nColsPop = len(sum([partCoordCols.tolist(),partVelCols.tolist(),partPbestCols.tolist(),  
                        [partFitPbestCols,partFitCurrCols,partFitLbestCols,partInertiaCols],
                        partLocalBestCols.tolist(),
                        [partFlagFitEvalCols,partFitEvalsCols]], []) )
    pop = np.zeros((psoParams['popsize'],nColsPop))

    # Best value found by the swarm over its history
    gbestVal = np.inf
    # Location of the best value found by the swarm over its history
    gbestLoc = 2 * np.ones((1,nDim)) # Init values for > (0,1)
    # Best value found by the swarm at the current iteration
    bestFitness = np.inf

    pop[:,partCoordCols] = np.random.rand(psoParams['popsize'],nDim)
    pop[:,partVelCols] = -pop[:,partCoordCols] + np.random.rand(psoParams['popsize'],nDim)
    pop[:,partPbestCols] = pop[:,partCoordCols]
    pop[:,partFitPbestCols]= np.inf
    pop[:,partFitCurrCols]=0
    pop[:,partFitLbestCols]= np.inf
    pop[:,partLocalBestCols] = 0
    pop[:,partFlagFitEvalCols]=1
    pop[:,partInertiaCols]=0
    pop[:,partFitEvalsCols]=0

    #Start PSO iterations ...
    for lpc_steps in tqdm(range(psoParams['maxSteps'])):
        #Evaluate particle fitnesses under ...
        if not psoParams['bndryCond']:
            #Invisible wall boundary condition
            fitnessValues = fitfuncHandle(pop[:,partCoordCols], returnxVec=0)
        else:
            #Special boundary condition (handled by fitness function)
            fitnessValues, pop[:,partCoordCols] = fitfuncHandle(pop[:,partCoordCols], returnxVec=1)
            
        #Fill pop matrix ...(for each partical; update FitCurr/FitEvals/FitPbest/Pbest)
        for k in range(psoParams['popsize']):
            pop[k, partFitCurrCols] = fitnessValues[k]
            computeOK = pop[k,partFlagFitEvalCols]
            funcCount = 1 if computeOK else 0
            pop[k,partFitEvalsCols] = pop[k,partFitEvalsCols] + funcCount
            if pop[k,partFitPbestCols] > pop[k,partFitCurrCols]:
                pop[k,partFitPbestCols] = pop[k,partFitCurrCols]
                pop[k,partPbestCols] = pop[k,partCoordCols]

        #Update gbest
        bestFitness, bestParticle = np.min(pop[:,partFitCurrCols]), np.argmin(pop[:,partFitCurrCols])
        if gbestVal > bestFitness:
            gbestVal = bestFitness
            gbestLoc = pop[bestParticle, partCoordCols]
            pop[bestParticle,partFitEvalsCols] = pop[bestParticle,partFitEvalsCols] + funcCount # Why ?

        #Local bests ...
        for k in range(psoParams['popsize']):
            #Get indices of neighborhood particles
            ringNbrs = np.roll(np.arange(psoParams['popsize']), shift=-k+1)[:psoParams['nbrhdSz']]

            #Get local best in neighborhood
            lbestPart = np.argmin(pop[ringNbrs,partFitCurrCols]) # get local indices in neighborhood
            lbestTruIndx = ringNbrs[lbestPart] # get global indices in pop
            lbestCurrSnr = pop[lbestTruIndx, partFitCurrCols]

            if lbestCurrSnr < pop[k,partFitLbestCols]:
                pop[k,partFitLbestCols] = lbestCurrSnr
                pop[k,partLocalBestCols] = pop[lbestTruIndx, partCoordCols]

        #Inertia decay (0.9~>0.2)
        inertiaWt = np.max([psoParams['dcLaw_a']-(psoParams['dcLaw_b']/psoParams['dcLaw_c'])*lpc_steps,
                            psoParams['dcLaw_d']])

        #Velocity updates ...
        for k in range(psoParams['popsize']):
            partInertia = pop[k,partInertiaCols] = inertiaWt
            chi1, chi2 = np.random.rand(nDim), np.random.rand(nDim)
            # PSO Dynamical Equation (Core)
            pop[k, partVelCols] = partInertia * pop[k,partVelCols] +\
                                psoParams['c1'] * (pop[k,partPbestCols] - pop[k,partCoordCols]) * chi1 +\
                                psoParams['c2'] * (pop[k,partLocalBestCols] - pop[k,partCoordCols])*chi2
            pop[k, partVelCols] = np.clip(pop[k, partVelCols], 
                                         -psoParams['max_velocity'], 
                                          psoParams['max_velocity'])
            pop[k,partCoordCols] = pop[k,partCoordCols] + pop[k,partVelCols]
            if np.any(pop[k,partCoordCols]<0) or np.any(pop[k,partCoordCols]>1):
                pop[k,partFitCurrCols]= np.inf
                pop[k,partFlagFitEvalCols]= 0
            else:
                pop[k,partFlagFitEvalCols]=1

        #Record extended output if needed
        if O==1:
            returnData['allBestFit'][lpc_steps] = gbestVal
        elif O==2:
            returnData['allBestLoc'][lpc_steps,:] = gbestLoc
        #Add more fields with additional case
        #statements

    actualEvaluations = np.sum(pop[:,partFitEvalsCols])

    #Prepare main output
    returnData['totalFuncEvals'] = actualEvaluations
    returnData['bestLocation'] = gbestLoc
    returnData['bestFitness'] = gbestVal
    return returnData

def crcbgenqcsig(dataX,r,m_c,tc,phi_c,mlz,y):
    # print(f"r:{r},mlz:{mlz},m_c:{m_c},phi:{phi_c}")
    r = (10 **  r) * 1e6 *  pc
    m_c =  (10 ** m_c) * M_sun
    mlz =(10 ** mlz) * M_sun
    # theta = c ** 3* (tc - dataX) / (5 * G * m_c)
    # print(f"r:{r/1e6/pc},mlz:{mlz/M_sun:.2e},m_c:{m_c/M_sun:.2e},phi:{phi_c/np.pi}")
    # print(f"theta:{theta}")
    # 时域上的信号波形
    # h = G * m_c / (c ** 2 * r) * theta ** (-1/4) * np.cos(2 * phi_c - 2 * theta ** (5 / 8))
    def generate_h_t(dataX,m_c,r,phi_c):
        if dataX < tc:
            theta_t = c **3 * (tc - dataX) / (5 * G * m_c)
            h = G * m_c/(c ** 2 * r) * theta_t**(-1/4) * np.cos(2 * phi_c - 2*theta_t ** (5/8))
        else :
            h = 0
        return h
    generate_h_t=np.frompyfunc(generate_h_t, 4, 1)
    h = generate_h_t(dataX,m_c,r,phi_c).astype(float)

    h_f = np.fft.rfft(h) # 时域到频域

    freqs = np.fft.rfftfreq(len(h),dataX[1]-dataX[0]) # 频率
    omega = 2 * np.pi * freqs 
    w = G *4 * mlz * omega / c ** 3 
    # F_geo = np.sqrt(1 + 1/y) - 1j * np.lib.scimath.sqrt(-1 + 1/ y) * np.exp(1j * w * 2 * y) # 透镜效应
    if y <1:
        F_geo = np.sqrt(1 + 1/y) - 1j*np.sqrt(-1 +1/y) * np.exp(1j * w * 2 *y)
    else :
        F_geo = np.sqrt(1 + 1/y)
    sigVec_f = h_f * F_geo # 给信号增加透镜效应（频域）
    sigVec = np.fft.irfft(sigVec_f) # 转换到时域
    # 转换到频域
    # print(f"sigvec:{sigVec}")
    # sigVec = 1 * sigVec / np.linalg.norm(sigVec)
    # print(f"r:{r/1e6/pc}, m_c:{m_c/M_sun}, tc:{tc}, phi_c:{phi_c}, mlz:{mlz/M_sun}, y:{y},f_geo:{F_geo},sigvec:{sigVec}")
    return sigVec



def glrtqcsig4pso(xVec, params, returnxVec=0):
    # rows: points
    # columns: coordinates of a point
    nVecs= xVec.shape[0]

    # storage for fitness values
    fitVal = np.zeros(nVecs)

    # Check for out of bound coordinates and flag them
    validPts = crcbchkstdsrchrng(xVec)
    # Set fitness for invalid points to infty
    fitVal[~validPts] = np.inf
    xVec[validPts,:] = s2rv(xVec[validPts,:], params)

    for lpc in range(nVecs):
        if validPts[lpc]:
        # Only the body of this block should be replaced for different fitness
        # functions
            x = xVec[lpc,:]
            fitVal[lpc] = ssrqc(x, params)

    # https://stackoverflow.com/questions/14147675/nargout-in-python
    #Return real coordinates if requested
    if returnxVec:
        return fitVal, xVec
    else:
        return fitVal

# Sum of squared residuals after maximizing over amplitude parameter
# def ssrqc(x, params):
#     phaseVec = x[0]*params['dataX'] + x[1]*params['dataXSq'] + x[2]*params['dataXCb']
#     qc = np.sin(2 * np.pi * phaseVec)
#     qc, _ = normsig4psd(qc, params['sampFreq'], params['psdPosFreq'], 1)
#     #We do not need the normalization factor, just the need template vector

#     #Compute fitness (Calculate inner product of data with template qc）
#     inPrd = innerprodpsd(params['dataY'], qc, params['sampFreq'], params['psdPosFreq'])
#     ssrVal = -(inPrd)**2
#     return ssrVal
def ssrqc(x, params):
    # Generate signal using crcbgenqcsig instead of phase model
    qc = crcbgenqcsig(params['dataX'], 
                      x[0],  # r
                      x[1],  # m_c
                      x[2],  # tc
                      x[3],  # phi_c
                      x[4],  # mlz
                      x[5],) # y
    # print(qc)
    # Normalize signal using PSD
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdPosFreq'], 1)
    # Compute fitness using inner product
    inPrd = innerprodpsd(params['dataY'], qc, params['sampFreq'], params['psdPosFreq'])
    ssrVal = -(inPrd)**2
    return ssrVal

def normsig4psd(sigVec, sampFreq, psdVec, snr):
    """
    PSD length must be commensurate with the length of the signal DFT 
    """
    nSamples = len(sigVec)
    kNyq = np.floor(nSamples/2) + 1
    assert len(psdVec) == kNyq, 'Length of PSD is not correct'
    # Norm of signal squared is inner product of signal with itself
    normSigSqrd = innerprodpsd(sigVec,sigVec,sampFreq,psdVec)
    # Normalization factor
    normFac = snr/np.sqrt(normSigSqrd)
    # Normalize signal to specified SNR
    normSigVec = normFac * sigVec
    return normSigVec, normFac

def innerprodpsd(xVec,yVec,sampFreq,psdVals):
    nSamples = len(xVec)
    assert len(yVec) == nSamples, 'Vectors must be of the same length'
    kNyq = np.floor(nSamples/2)+1
    assert len(psdVals) == kNyq, 'PSD values must be specified at positive DFT frequencies'

    # Why 'scipy.fftpack.fft'? 
    # See: https://iphysresearch.github.io/blog/post/signal_processing/fft/#efficiency-of-the-algorithms
    fftX = np.fft.fft(xVec)
    fftY = np.fft.fft(yVec)
    #We take care of even or odd number of samples when replicating PSD values
    #for negative frequencies
    negFStrt = 1 - np.mod(nSamples, 2)
    psdVec4Norm = np.concatenate((psdVals, psdVals[(kNyq.astype('int')-negFStrt)-1:0:-1]))
    dataLen = sampFreq * nSamples
    innProd = np.sum((1/dataLen) * (fftX / psdVec4Norm)*fftY.conj())
    # print(f"innProd:{innProd}")
    innProd = np.real(innProd)
    return innProd



def s2rv(xVec, params):
    return xVec * (np.asarray(params['rmax']) - np.asarray(params['rmin'])) + np.asarray(params['rmin'])

def crcbchkstdsrchrng(xVec):
    return np.array([False if np.any(xVec[lp]<0) or np.any(xVec[lp]>1) else True for lp in range(xVec.shape[0])])