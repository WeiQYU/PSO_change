import numpy as np
from scipy.fftpack import fft 
# Why this 'fft'? 
# See: https://iphysresearch.github.io/blog/post/signal_processing/fft/#efficiency-of-the-algorithms

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
# from numba import jit
# from numba import njit, prange

__all__ = ['crcbqcpsopsd', 
           'crcbpso', 
           'crcbgenqcsig', 
           'glrtqcsig4pso',
           'ssrqc',
           'normsig4psd',
           'innerprodpsd',
           's2rv',
           'crcbchkstdsrchrng']

# add the funcutions of Parallel
# from glrtqcsig4pso to crcbqcpsopsd(changed by ywq)
def glrtqcsig4pso_wrapper(x, inParams, returnxVec):
    return glrtqcsig4pso(x, inParams, returnxVec)


def run_single_pso(lpruns, inParams, nDim, psoParams):
    np.random.seed(lpruns)
    fHandle = lambda x, returnxVec: glrtqcsig4pso_wrapper(x, inParams, returnxVec)

    # 创建一个内层进度条
    with tqdm(total=psoParams['maxSteps'], desc=f"Run {lpruns + 1}", leave=False, position=lpruns + 1) as run_pbar:
        def update_pbar(step):
            run_pbar.update(1)

        # 修改 crcbpso 函数，添加回调以更新进度条
        result = crcbpso(fHandle, nDim, O=0, **psoParams, update_callback=update_pbar)

    return result

def crcbqcpsopsd(inParams, psoParams, nRuns):
    nSamples = len(inParams['dataX'])
    nDim = 3

    outStruct = [{} for _ in range(nRuns)]
    outResults = {
        'allRunsOutput': [],
        'bestRun': None,
        'bestFitness': None,
        'bestSig': np.zeros(nSamples),
        'bestQcCoefs': np.zeros(3),
        'bestSNR': None,
    }

    # 创建总进度条
    with tqdm(total=nRuns, desc="Overall Progress", position=0) as pbar:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_single_pso, lpruns, inParams, nDim, psoParams): lpruns for lpruns in range(nRuns)}

            for future in as_completed(futures):
                lpruns = futures[future]
                outStruct[lpruns] = future.result()
                pbar.update(1)

    # 准备输出
    fitVal = np.zeros(nRuns)
    for lpruns in range(nRuns):
        allRunsOutput = {
            'fitVal': 0,
            'qcCoefs': np.zeros(3),
            'estSig': np.zeros(nSamples),
            'totalFuncEvals': [],
            'snr': 0,
        }
        fitVal[lpruns] = outStruct[lpruns]['bestFitness']
        allRunsOutput['fitVal'] = fitVal[lpruns]

        fHandle = lambda x, returnxVec: glrtqcsig4pso_wrapper(x, inParams, returnxVec)
        _, qcCoefs = fHandle(outStruct[lpruns]['bestLocation'][np.newaxis, ...], returnxVec=1)
        allRunsOutput['qcCoefs'] = qcCoefs[0]
        estSig = crcbgenqcsig(inParams['dataX'], inParams['A'], qcCoefs[0])
        snr = claculateSNR(inParams,inParams['dataY'],estSig,inParams['sampFreq'],inParams['psdPosFreq']) # changed by ywq
        print("算法过程中snr:",snr)
        estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdPosFreq'], snr)
        estAmp = innerprodpsd(inParams['dataY'], estSig, inParams['sampFreq'], inParams['psdPosFreq'])
        estSig = estAmp * estSig

        allRunsOutput['estSig'] = estSig
        allRunsOutput['totalFuncEvals'] = outStruct[lpruns]['totalFuncEvals']
        outResults['allRunsOutput'].append(allRunsOutput)

    # 找到最佳运行
    bestRun = np.argmin(fitVal)
    outResults['bestRun'] = bestRun
    outResults['bestFitness'] = outResults['allRunsOutput'][bestRun]['fitVal']
    outResults['bestSig'] = outResults['allRunsOutput'][bestRun]['estSig']
    outResults['bestQcCoefs'] = outResults['allRunsOutput'][bestRun]['qcCoefs']
    outResults['bestSNR'] = outResults['allRunsOutput'][bestRun]['snr']

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

def crcbgenqcsig(dataX, snr, qcCoefs):
    phaseVec = qcCoefs[0]*dataX + qcCoefs[1]*dataX**2 + qcCoefs[2]*dataX**3
    sigVec = np.sin(2*np.pi*phaseVec)
    sigVec = snr*sigVec/np.linalg.norm(sigVec)
    return sigVec



def glrtqcsig4pso(xVec, params, returnxVec=0):
    # rows: points
    # columns: coordinates of a point
    nVecs, _ = xVec.shape

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
def ssrqc(x, params):
    phaseVec = x[0]*params['dataX'] + x[1]*params['dataXSq'] + x[2]*params['dataXCb']
    qc = np.sin(2 * np.pi * phaseVec)
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdPosFreq'], 1)
    #We do not need the normalization factor, just the need template vector

    #Compute fitness (Calculate inner product of data with template qc）
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
    fftX = fft(xVec)
    fftY = fft(yVec)
    #We take care of even or odd number of samples when replicating PSD values
    #for negative frequencies
    negFStrt = 1 - np.mod(nSamples, 2)
    psdVec4Norm = np.concatenate((psdVals, psdVals[(kNyq.astype('int')-negFStrt)-1:0:-1]))
    
    dataLen = sampFreq * nSamples
    innProd = np.sum((1/dataLen) * (fftX / psdVec4Norm)*fftY.conj())
    innProd = np.real(innProd)
    return innProd
# calculate SNR(changed by ywq) 
# if don't unified the length, it will wrong.but if unified,it will wrong too.
def claculateSNR(inparams,dataY,sigVec,sampFreq,psdVals):
    # unified the length 
    len_min = min(len(dataY), len(sigVec), len(inparams['noise']))
    inparams['noise'] = inparams['noise'][:len_min]
    sigVec = sigVec[:len_min]
    dataY = dataY[:len_min]
    llrNoise = innerprodpsd(inparams['noise'],sigVec,sampFreq,psdVals)
    llrData = innerprodpsd(dataY,sigVec,sampFreq,psdVals)
    estSNR = (np.mean(llrData) - np.mean(llrNoise)) / np.std(llrNoise)
    print(estSNR)
    return estSNR

def s2rv(xVec, params):
    return xVec * (np.asarray(params['rmax']) - np.asarray(params['rmin'])) + np.asarray(params['rmin'])

def crcbchkstdsrchrng(xVec):
    return np.array([False if np.any(xVec[lp]<0) or np.any(xVec[lp]>1) else True for lp in range(xVec.shape[0])])
