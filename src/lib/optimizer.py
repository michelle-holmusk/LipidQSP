import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import json
from multiprocessing import Pool
from collections     import deque
import time

import h5py
from copy import deepcopy
# from tqdm import tqdm

# config       = json.load(open('AmgenRR/config/config.json'))

class Optimizers:

    # @lD.log(logBase + '.Optimizers.__init__')
    def __init__(self, method, opt_kwargs={}, logModel_config=None):

        try:
            self.method          = method
            self.opt_kwargs      = opt_kwargs
            self.logModel_config = logModel_config

        except Exception as e:

            print('Unable to initialize Optimizers \n{}'.format(str))

    # @lD.log(logBase + '.Optimizers.fit')
    def fit(self, obj_fun, args, checkWeightRange=None):

        try:
            if   self.method == 'DifferentialEvolution':
                score, weights, hist = self.DE_optimize(obj_fun, args, checkWeightRange)
            elif self.method == 'AdamOptimizer':
                score, weights, hist = self.Adam_optimize(obj_fun, args, checkWeightRange)

            return score, weights, hist

        except Exception as e:

            print('Unable to fit \n{}'.format(str))

    # @lD.log(logBase + '.Optimizers.DE_optimize')
    def DE_optimize(self, obj_fun, args, checkWeightRange=None):

        try:
            de                   = Differential_Evolution(obj_fun=obj_fun, logModel_config=self.logModel_config, **self.opt_kwargs)
            score, weights, hist = de.optimize(args=args, checkWeightRange=checkWeightRange)

            return score, weights, hist

        except Exception as e:

            print('Unable to optimize based on Differential Evolution\n{}'.format(str(e)))

    # @lD.log(logBase + '.Optimizers.Adam_optimize')
    def Adam_optimize(self, obj_fun, args, checkWeightRange=None):

        try:
            adam                 = AdamDist(obj_fun=obj_fun, logModel_config=self.logModel_config, **self.opt_kwargs)
            # adam                 = Adam(obj_fun=obj_fun, logModel_config=self.logModel_config, **self.opt_kwargs)
            score, weights, hist = adam.optimize(args=args, checkWeightRange=checkWeightRange)

            return score, weights, hist

        except Exception as e:

            print('Unable to optimize based on Adam optimizer\n{}'.format(str(e)))

class Differential_Evolution(object):
    
    # @lD.log(logBase + '.Differential_Evolution.__init__')
    def __init__(self, obj_fun, eval_fun=None, npar=5, npool=50, bounds=[],
                    choice='linear', F=None, CR=0.9, strategy=1, 
                    maxiter=100, miniter=20, fmin=0, fmax=1.0, 
                    epsilon=5.0E-4, parallel=True,
                    service_based=False, nthreads=1,
                    early_stopping=20, verbose=True, output_folder='.', 
                    doEarlyStopping=True, dolimitbound='upper_lower', logModel_config=None,
                    init=None):

        try:
            self.obj_fun = obj_fun
            if eval_fun is None:
                eval_fun = obj_fun
            self.eval_fun = eval_fun
            self.parameter_number = npar
            self.population_size = npool
            self.bounds = bounds
            self.choice = choice
            self.F = F
            self.fb = np.array([fmin, fmax])
            self.CR = CR
            self.strategy = strategy
            self.maxiter = maxiter
            self.miniter = min(miniter, maxiter)
            self.population = np.zeros((npool, npar))
            self.scores = np.zeros(npool)
            self.best_member = np.zeros(npar)
            self.best_member_index = 0
            self.best_score = 1.0E6
            if init is not None:
                self.initial_guess = init
            else:
                self.initial_guess = []
            self.epsilon = epsilon
            self.parallel = parallel
            self.service_based = service_based
            self.nthreads = nthreads
            self.early_stopping = early_stopping
            self.verbose = verbose
            self.output_folder = output_folder
            self.doEarlyStopping = doEarlyStopping
            self.dolimitbound = dolimitbound

            # Added to be able to save the iterations
            if (self.output_folder != '.'):
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

        except Exception as e:

            print('Unable to initialize Differential Evolution\n{}'.format(str(e)))

    # def make_random_pool(self, arg):
    def make_random_pool(self):
        
        temp_score = -1
        
        while temp_score < 0:
            np.random.seed()
            random_values = np.random.random_sample(self.parameter_number)
            L = self.bounds[:,0]
            U = self.bounds[:,1]
            if self.choice == 'linear':
                new_pool_member = L + random_values*(U-L)
            elif self.choice == 'log':
                new_pool_member = (np.exp(np.log(L) + random_values*(np.log(U)-np.log(L))))
            # print("New random pool\t", new_pool_member)
            # temp_score = self.obj_fun(new_pool_member, *arg)
            temp_score = self.obj_fun(new_pool_member, *self.args)
            # print temp_score , ','.join(str(x) for x in new_pool_member)
            z = np.append(temp_score,new_pool_member)

        return z

    # def evolve(self, ii, p):
    def evolve(self, ii):
        
        np.random.seed()
        a1 = np.random.randint(self.population_size-1)
        a2 = np.random.randint(self.population_size-1)
        a3 = np.random.randint(self.population_size-1)
        if self.F is None:
            f_new = self.fb[0] + np.random.random_sample()*(self.fb[1] - self.fb[0])
        else:
            f_new = self.F
        pop_a1=(self.population[a1])
        pop_a2=(self.population[a2])
        pop_a3=(self.population[a3])
        pop_best=(self.best_member)

        if self.strategy == 2:
            # if f_new<0.5:
            if f_new<0.33:
                v = pop_a1 + f_new*(pop_a2 - pop_a3)
            # elif f_new>=0.5 and f_new<0.9:
            elif f_new>=0.33 and f_new<0.67:
                v = pop_a1 + 0.5*(f_new + 1)*(pop_a2 + pop_a3 - 2.0*pop_a1)
            else:
                v = pop_best + f_new*(pop_a1 - pop_a2)
        if self.strategy == 3:
            if np.random.rand()<0.5:
                f_new = np.random.normal(loc=0.5,scale=1.0)
            else:
                f_new = np.random.standard_cauchy()
            v    = pop_a1 + f_new*(pop_a2 - pop_a3)
        else:
            f_cr = np.random.random_sample()
            v    = self.population[ii] + f_cr*(pop_best - self.population[ii]) + f_new*(pop_a1 - pop_a2)
        
        if   self.dolimitbound == 'upper_lower':
            ui      = np.maximum(np.minimum(v ,self.bounds[:,1]), self.bounds[:,0])
        elif self.dolimitbound == 'upper':
            ui      = np.minimum(v, self.bounds[:, 1])
        elif self.dolimitbound == 'lower':
            ui      = np.maximum(v, self.bounds[:, 0])
        elif self.dolimitbound == 'none':
            ui      = v
            
        rand_xc = np.random.random_sample()
        mui_xc  = 0
        mpo_xc  = 0
        if (rand_xc < self.CR):
            mui_xc = 1
        else:
            mpo_xc = 1
        ui = self.population[ii]*mpo_xc + ui*mui_xc
        # print("evolve\t", ui)
        # tmp_score = self.obj_fun(ui, *p)
        tmp_score = self.obj_fun(ui, *self.args)
        if tmp_score < 0:
            # z = self.make_random_pool(p)
            z = self.make_random_pool()
            ui = z[1:]
            tmp_score = z[0]
            # tmp_score = 1e10
        X = np.append(tmp_score,ui)
        
        return X

    # def evolve_intermediate(self, args):
    def evolve_intermediate(self, i):
        
        # i, self, p = args
        
        # return self.evolve(i, p)
        return self.evolve(i)

    # def make_random_pool_intermediate(self, args):
    def make_random_pool_intermediate(self, i):

        # i, self, arg = args
        
        # return self.make_random_pool(arg)
        return self.make_random_pool()

    # @lD.log(logBase + '.Differential_Evolution.optimize')
    # def optimize(self, args, checkWeightRange=None):
    def optimize(self, args, checkWeightRange=None, pool=None):

        try:
            t0           = time.time()
            # arg0         = deepcopy(args)
            # index        = [(i, self, arg0) for i in range(self.population_size)]
            self.args    = deepcopy(args)
            index        = [(i) for i in range(self.population_size)]

            if self.parallel:  # Parallelize if requested
                if pool is None:
                    if self.service_based:
                        pool = Pool(self.nthreads)
                    else:
                        pool = Pool(processes=8)
                Y = np.asarray(pool.map(self.make_random_pool_intermediate, index))
                # Y = np.array([m for m in pool.imap(self.make_random_pool_intermediate, index)])
                # pool.close()
            else:  # Run serially
                Y = np.asarray(list(map(self.make_random_pool_intermediate, index)))

            print('Done first iter')

            self.scores = Y[:, 0]
            self.population = Y[:, 1:]
            if len(self.initial_guess) > 0:
                self.population[0] = self.initial_guess
                # self.scores[0] = self.obj_fun(self.initial_guess, *arg0)
                self.scores[0] = self.obj_fun(self.initial_guess, *self.args)
            count     = 1
            converged = False
            self.iter_count = []
            self.iter_score = []
            self.weightsLog  = []
            self.weightsLog2 = []
            self.best_score = np.nanmin(self.scores)
            self.best_member_index = np.nanargmin(self.scores)
            self.good_member_indexes = np.argsort(self.scores)
            self.best_member = self.population[self.best_member_index]
            # self.best_eval_score = self.eval_fun(self.best_member, *arg0)

            t1 = time.time()
            CV = (
                100.0 *
                np.std(self.population, axis=0) / np.mean(self.population, axis=0)
            )
            CV_max     = np.max(CV)
            time_spent = t1-t0
            
            if self.verbose:
                print('Time elapsed:', (time_spent))
                print('Iteration = ', count),
                print('DE Score =', self.best_score)
                if checkWeightRange is None:
                    # print('First 50 Params: ', self.best_member[:50])
                    print('Best Params: ', self.best_member)
                else:
                    listOfGoodWeights = [self.population[index] 
                                            for i, index in enumerate(self.good_member_indexes) 
                                                if i < 10]
                    self.weightsLog2.append(  checkWeightRange( listOfGoodWeights )  )
                    weights_by_group = checkWeightRange([self.best_member])
                    for key in weights_by_group.keys():
                        print( '{} min: {:.3f}, max: {:.3f}'.format(key, weights_by_group[key].min(), weights_by_group[key].max()) )
                    self.weightsLog.append(weights_by_group)
                print('CV =', CV_max, '\n')
                # Save output to a text file
                print("\t".join([str(i) for i in self.best_member.tolist()]), file=open(os.path.join(self.output_folder, 'DE_iterations_temp.txt'), 'a'), end='\n')
                print(time_spent, file=open(os.path.join(self.output_folder, 'DE_iterations_timeSpent_temp.txt'), 'a'), end='\n')
                print(self.best_score, file=open(os.path.join(self.output_folder, 'DE_iterations_bestScore_temp.txt'), 'a'), end='\n')


            self.iter_count.append(count)
            self.iter_score.append(self.best_score)

            while not converged and count < self.maxiter:
                count += 1
                t0    = time.time()
                # arg0  = deepcopy(args)
                # index = [(i, self, arg0) for i in range(self.population_size)]
                index        = [(i) for i in range(self.population_size)]

                if self.parallel:
                    if pool is None:
                        if self.service_based:
                            pool = Pool(self.nthreads)
                        else:
                            pool = Pool(processes=8)
                    X = np.asarray(pool.map(self.evolve_intermediate, index))
                    # X = np.array([m for m in pool.imap(self.evolve_intermediate, index)])
                    # pool.close()

                else:
                    X = np.asarray(list(map(self.evolve_intermediate, index)))
                # print('after evolve, {:.3f}'.format(time.time() - t0))

                tmp_pop    = X[:, 1:]
                tmp_scores = X[:, 0]
                B = np.less(X[:, 0], self.scores)
                A = np.transpose(np.array([B, ]*self.parameter_number))
                self.population = self.population*(1 - A) + A*tmp_pop
                self.scores = self.scores*(1 - B) + B*tmp_scores
                
                self.best_score = np.nanmin(self.scores)
                self.best_member_index = np.nanargmin(self.scores)
                self.good_member_indexes = np.argsort(self.scores)
                self.best_member = self.population[self.best_member_index]
                # self.best_eval_score = self.eval_fun(self.best_member, *arg0)

                self.iter_count.append(count)
                self.iter_score.append(self.best_score)
                
                t1 = time.time()
                CV = (
                    100.0 *
                    np.std(self.population, axis=0) / np.mean(self.population, axis=0)
                )
                CV_max     = np.max(CV)
                time_spent = t1-t0
                # print('after miscellaneous, {:.3f}'.format(time.time() - t0))
                
                if self.verbose:
                    print('Time elapsed:', (time_spent))
                    print('Iteration = ', count),
                    print('DE Score =', self.best_score)
                    if checkWeightRange is None:
                        # print('First 50 Params: ', self.best_member[:50])
                        print('Best Params: ', self.best_member)
                        # Save output to a text file
                        print("\t".join([str(i) for i in self.best_member.tolist()]), file=open(os.path.join(self.output_folder, 'DE_iterations_temp.txt'), 'a'), end='\n')
                        print(time_spent, file=open(os.path.join(self.output_folder, 'DE_iterations_timeSpent_temp.txt'), 'a'), end='\n')
                        print(self.best_score, file=open(os.path.join(self.output_folder, 'DE_iterations_bestScore_temp.txt'), 'a'), end='\n')
                    else:
                        listOfGoodWeights = [self.population[index] 
                                                for i, index in enumerate(self.good_member_indexes) 
                                                    if i < 10]
                        self.weightsLog2.append(  checkWeightRange( listOfGoodWeights )  )
                        weights_by_group = checkWeightRange([self.best_member])
                        for key in weights_by_group.keys():
                            print( '{} min: {:.3f}, max: {:.3f}'.format(key, weights_by_group[key].min(), weights_by_group[key].max()) )
                        self.weightsLog.append(weights_by_group)
                    print('CV =', CV_max, '\n')

                if count > self.miniter:
                    delta_score = np.fabs(self.iter_score[-1] - self.iter_score[-self.early_stopping])
                    if (self.doEarlyStopping and (delta_score < self.epsilon)) or CV_max < 1.0:
                        converged = True
            
            if pool is not None:
                pool.close()

            return self.best_score, self.best_member, (self.iter_count, 
                        self.iter_score, self.weightsLog, self.weightsLog2)

        except Exception as e:

            print('Unable to optimize under Differential Evolution\n{}'.format(str(e)))

class AdamDist:

    # @lD.log(logBase + '.AdamDist.__init__')
    def __init__(self, obj_fun, npar, initWeights=None, 
        pertScale=1e-3, pertSize=3, # Parameters for the jacobian calculations
        learningRate=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, # Adam params
        wt = 1e-4, # AdamW parameters (weight decay)
        useAMSGrad  = False, # AMSGrad parameters 
        parallelize = False,
        nSteps=100, mStop=1e-6, vStop=1e-6, optStop=1e-8, nStop=10, # Stopping criterion
        logModel_config=None, freqLog=50):
        '''Adam optimizer for numpy
        
        This is an Adam optimizer that will be used for optimizing (minizing) the 
        objective function that is provided. The jacobian is calculated numerically. 
        This is done slightly differently from normal numerical differentiation 
        techniques. 
        
        Parameters
        ----------
        obj_fun : {function}
            This is the objective function that you are trying to minimize. This function
            will take two parameters. 
                - A 1D numpy array as the weights that we are trying to optimize
                - other parameters that we are not trying to optimize. 
        npar : {int}
            The number of parameters to be optimized. This will generate the required weights
            to track internally. This will be ignored if the ``initWeights`` parameter is 
            present.
        initWeights : {numpy ndarray}, optional
            Initial weights in one dimentsion (the default is None, which will result in the
            generation of random initial weights close to zero)
        pertScale : {float}, optional
            The size of the perturbation that will be used for generating points for each of the
            weights. The larger the size, the bigger the size of the perturbationa about the current
            weights used for calculating the slope around the weights (the default is 1e-3)
        pertSize : {int}, optional
            The number of points you want to use for generating data around the current weight that 
            will be used for the calculation of the slope. The greater this number, the more this
            this function will be able to generalzie and avoid noise. However, this is also going to
            increase the conputational time. Rememebr that you will at least need 2 points to calculate 
            a slope. Anything less than this and this function will fail. (the default is 3)
        learningRate : {float}, optional
            The learning rate of the optimizer. The higher the learning rate, the faster this is going to
            be able to get to the solution. Rememebr that high values of learning rates will lead to 
            instability, and thus use this with caution. (the default is 0.001, which is the default that is
            provided by the original paper for Adam)
        beta1 : {float}, optional
            The rate with which the momentum term of the optimizer should be changed (the default is 0.9, 
            which is the default that is provided by the original paper for Adam)
        beta2 : {number}, optional
            The rate at which the velocity term is changed in each iteration (the default is 0.999, which is 
            the default that is provided by the original paper for Adam)
        epsilon : {float}, optional
            Small number that is used to make sure that the value of the denominator does not ever reach zero
            to prevent overflow problems (the default is 1e-8, which is the default that is provided by the 
            original paper for Adam)
        wt : {float}, optional
            The weight decay that is used for regularization. You can either use a regularization term within
            your objective function, or simply use it here. To turn it off, just set this value to ``0``.(the 
            default is 1e-4)
        useAMSGrad : {bool}, optional
            In the case that the optimizer is being trainedin batches, this sometimes helps with the problem
            of significant velocity reduction due to the vagrancies of the current batch of data used for 
            training. You may consider it if there is significant diversion in the data used during batch 
            optimization. (the default is False, which results in this feature not being used.)
        parallelize : {bool}, optional
            Whether to calculate the jacobian in a parallelized manner. This will allow the results to be 
            calculated faster, but uses a significant amount of overhead for the parallel processing. Hence
            this is only ideal when the number of parameters are fairly large (the default is ``False``, which 
            turns off parallelization)
        nSteps : {int}, optional
            This is the maximum number of steps for which the optimizer will be run (the default is 100)
        mStop : {float}, optional
            Early stopping criterion. This is implemented that will allow the optimizer to stop early if 
            the optimizer is not improving any more (the default is 1e-4, which is just a random number. This
            should be adapted for your own problems). Note that both this and the velocity term has to be
            simultaneously corrected for the early stopping to be activated.
        vStop : {float}, optional
            Early stopping criterion for the velocity term. This will allow the optimizer to stop early if the
            optimizer velocity is close to zero. (the default is 1e-4, which is really a value thas has been 
            randomly generated and thus you should optimize this for your process.)
        optStop : {float}, optional
            Early stopping criterion for the error term. This will allow the optimizer to stop early if the
            optimizer velocity is close to zero. (the default is 1e-4, which is really a value thas has been 
            randomly generated and thus you should optimize this for your process.)
        nStop : {int}, optimal
            Number of things that we need to average over for the differences to be calculated
        '''
        
        try:
            self.obj_func  = obj_fun
            self.N         = npar

            # Parameters for the adam optimizer
            self.lr        = learningRate
            self.beta1     = beta1
            self.beta2     = beta2
            self.epsilon   = epsilon

            # parameters for the AdamW optimizer
            self.wt        = wt

            # use AMSGrad if necessary
            self.useAMSGrad = useAMSGrad
            self.vHatOld    = np.zeros(self.N)

            self.m = np.zeros(self.N) # Initial momentum and velocity components
            self.v = np.zeros(self.N)
            self.t = 0

            # Paraemters for the stopping criterion
            self.nSteps      = nSteps
            
            self.mStop       = mStop
            self.vStop       = vStop
            self.optStop     = optStop
            self.nStop       = nStop

            self.logModel_config = logModel_config
            self.freqLog         = freqLog

            self.mStopArr    = deque([], maxlen=nStop)
            self.vStopArr    = deque([], maxlen=nStop)
            self.optStopArr  = deque([], maxlen=nStop)
            self.goodweights = deque([], maxlen=nStop)

            self.parallelize = parallelize

            if initWeights is None:
                self.weights = np.random.rand(npar)*0.1
            else:
                self.weights = initWeights

            # weight perturbation parameters
            self.pertScale = pertScale
            self.pertSize  = pertSize

            return

        except Exception as e:

            print('Unable to initialize adam optimizer \n{}'.format(str(e)))

    def calcIthJac(self, args_i):
        '''calculate the jacobian for the ith arguments
        
        This function calculates the Jacobian for the ith weight
        in the array. 
        
        Parameters
        ----------
        args_i : {tuple}
            This is the combination of the arguments (i.e. `args`) that
            is to be furnished to the optimizer function, and the index
            of the weight matrix that is to be optimized.
        
        Returns
        -------
        number
            The value of the partial derivative of the function wrt the ith
            weight
        '''

        jac = 0

        try:
            args, i     = args_i

            oldWs       = np.array([self.weights.copy() for _ in range(self.pertSize)])
            w           = oldWs[0, i]
            newWs       = np.random.normal(w, self.pertScale, self.pertSize)
            oldWs[:, i] = newWs

            newVals     = np.array([self.obj_func(m, *args) for m in oldWs])
            jac         = np.polyfit(newWs, newVals, 1)[0]

        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception as e:
            print(f'Problem with {i}th jacobian: {e}')
            print(newWs, newVals, self.pertScale, self.pertSize, w)
            print(oldWs)
            
            return 0

        return jac

    # @lD.log(logBase + '.AdamDist.calcJac')
    def calcJac(self, args):
        '''The jacobian
        
        This is the value of the partial derivatives with respect to
        each of the weights in the weight vector. This is either 
        calculated in parallel or in series, depending upon how the
        optimizer is initially optimized. 
        
        Parameters
        ----------
        args : {tuple}
            A tuple of arguments that needs to be passed to the objective function
            along with the weights that are to be optimized.
        
        Returns
        -------
        numpy.ndarray
            This returns a numpy array that is the same shape as the weights array.
            Each value is the result of a partial derivative of the objective 
            function with respect to the weights.
        '''

        try:
            jacs  = []

            if not self.parallelize:
                oldWs = np.array([self.weights.copy() for _ in range(self.pertSize)])
                for i, w in enumerate(self.weights):
                    newWs  = np.random.normal(w, self.pertScale, self.pertSize)
                    tempWs = oldWs.copy()
                    tempWs[:, i] = newWs

                    newVals = np.array([self.obj_func(m, *args) for m in tempWs])

                    slope = np.polyfit(newWs, newVals, 1)[0]
                    jacs.append(slope)

            else:
                pool   = Pool()
                args_i = [(args, i) for i in range(self.N)]
                for j in pool.imap(self.calcIthJac, args_i):
                    jacs.append(j)

                pool.close()

            jacs = np.array(jacs)

            return jacs

        except Exception as e:

            print('Unable to calculate jacobian for all parameters \n{}'.format(str(e)))

    # @lD.log(logBase + '.AdamDist.setWeights')
    def setWeights(self, w):
        '''set the current weights of the optimizer
        
        This is useful when you are starting with a set of weights
        that have already been pretrained using another optimizer.
        
        Parameters
        ----------
        w : {numpy.ndarray}
            A set of weights that this new optimizer should use. Note
            that it is illegal to supply a set of weights that have a
            different shape than the current shape.
        '''

        try:
            assert self.weights.shape == w.shape, 'The shape if w [{}] must match the shape of the current weights [{}]'.format( w.shape, self.weights.shape )

            self.weights = w
            return

        except Exception as e:

            print('Unable to set weights \n{}'.format(str(e)))

    # @lD.log(logBase + '.AdamDist.step')
    def step(self, args):
        '''Take one step towards the optima
        
        This function takes a single step toward the optimum solution
        according to the AdamW algorithm. 
        
        Parameters
        ----------
        args : {tuple}
            These are the set of arguments that are to be supplied to the
            objective function that will allow the objective function to
            be calculated.
        '''
        
        try:

            jacs = self.calcJac(args)
            
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1)*jacs
            self.v = self.beta2 * self.v + (1 - self.beta1)*jacs**2

            mHat = self.m / ( 1 - self.beta1 ** self.t )
            vHat = self.v / ( 1 - self.beta2 ** self.t )

            if self.useAMSGrad:
                vHat = np.maximum( vHat, self.vHatOld )
                self.vHatOld = vHat

            self.weights -= self.lr * self.wt * self.weights # weight decay
            self.weights -= self.lr * mHat / ( self.epsilon + vHat )

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print('Error: Some problem with the step function: {}'.format(str(e)))
            return

        return

    # @lD.log(logBase + '.AdamDist.optimize')
    def optimize(self, args, checkWeightRange=None):
        '''Take a set of steps toward the optimum
        
        This function will take a predetermined number of steps
        toward the optimum solution. This has several stopping 
        criterion that may be employed to check whether the optimizer
        should have early stopping
        
        Parameters
        ----------
        args : {tuple}
            These are the set of arguments that are to be supplied to the
            objective function that will allow the objective function to
            be calculated.
        
        Returns
        -------
        numpy.ndarray
            The weights as a result of the optimization process
        '''

        try:
            print('Doing an optimization ...')
            pbar            = tqdm(range(self.nSteps))
            self.iter_count = []
            self.iter_score = []
            self.weightsLog = {'m': [], 'v': [], 'weights': []} # Add saving of weights
            # self.weightsLog = {'m': [], 'v': []}

            for i in pbar:
                try:
                    self.step(args)
                    self.best_score = self.obj_func(self.weights, *args)

                    self.iter_count.append(i)
                    self.iter_score.append(self.best_score)

                    self.weightsLog['m'].append(self.m)
                    self.weightsLog['v'].append(self.v)
                    self.weightsLog['weights'].append(self.weights) # Add saving of weights

                    pbar.set_description('score: {:.3f} | m: ({:.1E}, {:.1E}) | v: ({:.1E}, {:.1E})'.format(
                                                    self.best_score, self.m.min(), self.m.max(), 
                                                    self.v.min(), self.v.max()))

                    self.mStopArr.append(self.m.max())
                    self.vStopArr.append(self.v.max())
                    self.optStopArr.append(self.best_score)
                    self.goodweights.append(self.weights)

                    if self.logModel_config is not None and i != 0 and (i + 1) % self.freqLog == 0:

                        outputFolder    = self.logModel_config['outputFolder']
                        first_on_Var    = self.logModel_config['first_on_Var']
                        second_on_Var   = self.logModel_config['second_on_Var']
                        preTrainedModel = self.logModel_config['preTrainedModel']
                        filename        = preTrainedModel.split('.')[0] + '_{:05d}'.format(i + 1) + '.' + preTrainedModel.split('.')[1]

                        with h5py.File( os.path.join(outputFolder, filename) ) as h5py_file:
                            h5py_file.create_dataset('weights',       data=self.weights,     dtype=np.float64)
                            h5py_file.create_dataset('score',         data=self.best_score,  dtype=np.float64)
                            h5py_file.create_dataset('first_on_Var',  data=first_on_Var)
                            h5py_file.create_dataset('second_on_Var', data=second_on_Var)

                        pbar.write('saving checkpoint.. {}'.format(os.path.join(outputFolder, filename)))

                    if i <= self.nStop - 1:
                        continue

                    if (np.abs(np.mean(self.mStopArr)) < self.mStop) or \
                       (np.abs(np.mean(self.vStopArr)) < self.vStop) or \
                       (np.abs(np.mean(self.optStopArr) - self.best_score) < self.optStop) :
                       print('Early stopping here ...')

                       break

                except (KeyboardInterrupt, SystemExit):
                    print('Trying to stop the program ...')
                    return self.weights
                except Exception as e:
                    print(f'Error: Some problem with the step function: {e}')
                    return

            bestweightIndex   = np.nanargmin(self.optStopArr)

            self.best_score   = self.optStopArr[bestweightIndex]
            self.best_weights = self.goodweights[bestweightIndex]

            return self.best_score, self.best_weights, (self.iter_count, self.iter_score, self.weightsLog)

        except Exception as e:

            print('Unable to optimize using adam optimizer \n{}'.format(str(e)))

class Adam:

    # @lD.log(logBase + '.Adam.__init__')
    def __init__(self, obj_fun, npar, mask=None, initWeights=None, 
        pertScale=1e-3, pertSize=3, # Parameters for the jacobian calculations
        learningRate=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, # Adam params
        wt = 1e-4, # AdamW parameters (weight decay)
        useAMSGrad  = False, # AMSGrad parameters 
        parallelize = False, useFastJac = False,
        nSteps=100, mStop=1e-6, vStop=1e-6, optStop=1e-8, nStop=10, # Stopping criterion
        logModel_config=None, freqLog=50): 
        '''Adam optimizer for numpy
        
        This is an Adam optimizer that will be used for optimizing (minizing) the 
        objective function that is provided. The jacobian is calculated numerically. 
        This is done slightly differently from normal numerical differentiation 
        techniques. 
        
        Parameters
        ----------
        obj_fun : {function}
            This is the objective function that you are trying to minimize. This function
            will take two parameters. 
                - A 1D numpy array as the weights that we are trying to optimize
                - other parameters that we are not trying to optimize. 
        npar : {int}
            The number of parameters to be optimized. This will generate the required weights
            to track internally. This will be ignored if the ``initWeights`` parameter is 
            present.
        mask : {numpy.ndarray or None}, optional
            This is a binary mask that has the same shape as the shape of the weights. All weights
            that need to be optimized should be set to ``1``, and the rest should be set to zero.
            Only the ones that have a ``1`` in the mask will be updated. Note that this is only
            true for the case wherein an array is supplied. In case a ``None`` is provided, all
            weights will be used for the jacobian calculation.
        initWeights : {numpy ndarray}, optional
            Initial weights in one dimentsion (the default is None, which will result in the
            generation of random initial weights close to zero)
        pertScale : {float}, optional
            The size of the perturbation that will be used for generating points for each of the
            weights. The larger the size, the bigger the size of the perturbationa about the current
            weights used for calculating the slope around the weights (the default is 1e-3)
        pertSize : {int}, optional
            The number of points you want to use for generating data around the current weight that 
            will be used for the calculation of the slope. The greater this number, the more this
            this function will be able to generalzie and avoid noise. However, this is also going to
            increase the conputational time. Rememebr that you will at least need 2 points to calculate 
            a slope. Anything less than this and this function will fail. (the default is 3)
        learningRate : {float}, optional
            The learning rate of the optimizer. The higher the learning rate, the faster this is going to
            be able to get to the solution. Rememebr that high values of learning rates will lead to 
            instability, and thus use this with caution. (the default is 0.001, which is the default that is
            provided by the original paper for Adam)
        beta1 : {float}, optional
            The rate with which the momentum term of the optimizer should be changed (the default is 0.9, 
            which is the default that is provided by the original paper for Adam)
        beta2 : {number}, optional
            The rate at which the velocity term is changed in each iteration (the default is 0.999, which is 
            the default that is provided by the original paper for Adam)
        epsilon : {float}, optional
            Small number that is used to make sure that the value of the denominator does not ever reach zero
            to prevent overflow problems (the default is 1e-8, which is the default that is provided by the 
            original paper for Adam)
        wt : {float}, optional
            The weight decay that is used for regularization. You can either use a regularization term within
            your objective function, or simply use it here. To turn it off, just set this value to ``0``.(the 
            default is 1e-4)
        useAMSGrad : {bool}, optional
            In the case that the optimizer is being trainedin batches, this sometimes helps with the problem
            of significant velocity reduction due to the vagrancies of the current batch of data used for 
            training. You may consider it if there is significant diversion in the data used during batch 
            optimization. (the default is False, which results in this feature not being used.)
        parallelize : {bool}, optional
            Whether to calculate the jacobian in a parallelized manner. This will allow the results to be 
            calculated faster, but uses a significant amount of overhead for the parallel processing. Hence
            this is only ideal when the number of parameters are fairly large (the default is ``False``, which 
            turns off parallelization)
        nSteps : {int}, optional
            This is the maximum number of steps for which the optimizer will be run (the default is 100)
        mStop : {float}, optional
            Early stopping criterion. This is implemented that will allow the optimizer to stop early if 
            the optimizer is not improving any more (the default is 1e-4, which is just a random number. This
            should be adapted for your own problems). Note that both this and the velocity term has to be
            simultaneously corrected for the early stopping to be activated.
        vStop : {float}, optional
            Early stopping criterion for the velocity term. This will allow the optimizer to stop early if the
            optimizer velocity is close to zero. (the default is 1e-4, which is really a value thas has been 
            randomly generated and thus you should optimize this for your process.)
        optStop : {float}, optional
            Early stopping criterion for the error term. This will allow the optimizer to stop early if the
            optimizer velocity is close to zero. (the default is 1e-4, which is really a value thas has been 
            randomly generated and thus you should optimize this for your process.)
        nStop : {int}, optimal
            Number of things that we need to average over for the differences to be calculated
        '''
        
        try:

            self.obj_func  = obj_fun
            self.N         = npar
            self.mask      = mask

            # Parameters for the adam optimizer
            self.lr        = learningRate
            self.beta1     = beta1
            self.beta2     = beta2
            self.epsilon   = epsilon

            # parameters for the AdamW optimizer
            self.wt        = wt

            # use AMSGrad if necessary
            self.useAMSGrad = useAMSGrad
            self.vHatOld    = np.zeros(self.N)

            self.m = np.zeros(self.N) # Initial momentum and velocity components
            self.v = np.zeros(self.N)
            self.t = 0

            # Paraemters for the stopping criterion
            self.nSteps      = nSteps
            
            self.mStop       = mStop
            self.vStop       = vStop
            self.optStop     = optStop
            self.nStop       = nStop
            
            self.logModel_config = logModel_config
            self.freqLog         = freqLog

            self.mStopArr    = deque([], maxlen=nStop)
            self.vStopArr    = deque([], maxlen=nStop)
            self.optStopArr  = deque([], maxlen=nStop)
            self.goodweights = deque([], maxlen=nStop)

            self.parallelize = parallelize
            self.useFastJac  = useFastJac

            if initWeights is None:
                self.weights = np.random.rand(npar)*0.1
            else:
                self.weights = initWeights

            # weight perturbation parameters
            self.pertScale = pertScale
            self.pertSize  = pertSize

            return

        except Exception as e:

            print('Unable to initialize Adam optimizer \n{}'.format(str(e)))

    def calcIthJac(self, args_i):
        '''calculate the jacobian for the ith arguments
        
        This function calculates the Jacobian for the ith weight
        in the array. 
        
        Parameters
        ----------
        args_i : {tuple}
            This is the combination of the arguments (i.e. `args`) that
            is to be furnished to the optimizer function, and the index
            of the weight matrix that is to be optimized.
        
        Returns
        -------
        number
            The value of the partial derivative of the function wrt the ith
            weight
        '''

        jac = 0

        try:

            args, i = args_i

            if (self.mask is not None) and ( self.mask[i] == 0 ):
                return 0

            oldWs = np.array([self.weights.copy() for _ in range(self.pertSize)])
            w     = oldWs[0, i]
            newWs = np.random.normal(w, self.pertScale, self.pertSize)
            oldWs[:, i] = newWs

            newVals = np.array([self.obj_func(m, *args) for m in oldWs])
            jac = np.polyfit(newWs, newVals, 1)[0]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(f'Problem with Jth jacobian: {e}')
            return 0

        return jac

    # @lD.log(logBase + '.Adam.calcJac')
    def calcJac(self, args):
        '''The jacobian
        
        This is the value of the partial derivatives with respect to
        each of the weights in the weight vector. This is either 
        calculated in parallel or in series, depending upon how the
        optimizer is initially optimized. 
        
        Parameters
        ----------
        args : {tuple}
            A tuple of arguments that needs to be passed to the objective function
            along with the weights that are to be optimized.
        
        Returns
        -------
        numpy.ndarray
            This returns a numpy array that is the same shape as the weights array.
            Each value is the result of a partial derivative of the objective 
            function with respect to the weights.
        '''

        try:
            jacs  = []

            if not self.parallelize:
                oldWs = np.array([self.weights.copy() for _ in range(self.pertSize)])
                for i, w in enumerate(self.weights):

                    if (self.mask is not None) and (self.mask[i] == 0):
                        jacs.append(0)
                        continue

                    newWs  = np.random.normal(w, self.pertScale, self.pertSize)
                    tempWs = oldWs.copy()
                    tempWs[:, i] = newWs

                    newVals = np.array([self.obj_func(m, *args) for m in tempWs])

                    slope = np.polyfit(newWs, newVals, 1)[0]
                    jacs.append(slope)

            else:
                pool   = Pool()
                args_i = [(args, i) for i in range(self.N)]
                for j in pool.imap(self.calcIthJac, args_i):
                    jacs.append(j)

                pool.close()

            jacs = np.array(jacs)

            return jacs

        except Exception as e:

            print('Unable to calculate jacobian \n{}'.format(str(e)))

    # @lD.log(logBase + '.Adam.fastJac')
    def fastJac(self, args):

        try:
            jacs    = []
            oldWs   = np.array([self.weights.copy() for _ in range(self.pertSize)])
            newWs   = np.random.normal(0, self.pertScale, oldWs.shape)
            tempWs  = oldWs + newWs
            newVals = np.array([self.obj_func( tempWs[m], *args) for m in range(self.pertSize)  ])

            for i, w in enumerate(self.weights):

                if (self.mask is not None) and (self.mask[i] == 0):
                    jacs.append(0)
                    continue

                xVals = newWs[:, i]
                slope = np.polyfit(xVals, newVals, 1)[0]
                jacs.append(slope)

            return np.array(jacs)

        except Exception as e:

            print('Unable to do fast jacobian \n{}'.format(str(e)))

    # @lD.log(logBase + '.Adam.setWeights')
    def setWeights(self, w):
        '''set the current weights of the optimizer
        
        This is useful when you are starting with a set of weights
        that have already been pretrained using another optimizer.
        
        Parameters
        ----------
        w : {numpy.ndarray}
            A set of weights that this new optimizer should use. Note
            that it is illegal to supply a set of weights that have a
            different shape than the current shape.
        '''

        try:
            assert self.weights.shape == w.shape, 'The shape if w [{}] must match the shape of the current weights [{}]'.format( w.shape, self.weights.shape )

            self.weights = w
            return

        except Exception as e:

            print('Unable to set weights \n{}'.format(str(e)))

    # @lD.log(logBase + '.Adam.step')
    def step(self, args):
        '''Take one step towards the optima
        
        This function takes a single step toward the optimum solution
        according to the AdamW algorithm. 
        
        Parameters
        ----------
        args : {tuple}
            These are the set of arguments that are to be supplied to the
            objective function that will allow the objective function to
            be calculated.
        '''
        
        try:

            if self.useFastJac:
                jacs = self.fastJac(args)
            else:
                jacs = self.calcJac(args)
            
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1)*jacs
            self.v = self.beta2 * self.v + (1 - self.beta1)*jacs**2

            mHat = self.m / ( 1 - self.beta1 ** self.t )
            vHat = self.v / ( 1 - self.beta2 ** self.t )

            if self.useAMSGrad:
                vHat = np.maximum( vHat, self.vHatOld )
                self.vHatOld = vHat

            self.weights -= self.lr * self.wt * self.weights # weight decay
            self.weights -= self.lr * mHat / ( self.epsilon + vHat )

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(f'Error: Some problem with the step function: {e}')

        return

    # @lD.log(logBase + '.Adam.optimize')
    def optimize(self, args, checkWeightRange=None):
        '''Take a set of steps toward the optimum
        
        This function will take a predetermined number of steps
        toward the optimum solution. This has several stopping 
        criterion that may be employed to check whether the optimizer
        should have early stopping
        
        Parameters
        ----------
        args : {tuple}
            These are the set of arguments that are to be supplied to the
            objective function that will allow the objective function to
            be calculated.
        
        Returns
        -------
        numpy.ndarray
            The weights as a result of the optimization process
        '''

        try:
            print('Doing an optimization ...')
            pbar            = tqdm(range(self.nSteps))
            self.iter_count = []
            self.iter_score = []
            self.weightsLog = {'m': [], 'v': []}

            for i in pbar:
                try:
                    self.step(args)
                    self.best_score = self.obj_func(self.weights, *args)

                    self.iter_count.append(i)
                    self.iter_score.append(self.best_score)

                    self.weightsLog['m'].append(self.m)
                    self.weightsLog['v'].append(self.v)

                    pbar.set_description('score: {:.3f} | m: ({:.1E}, {:.1E}) | v: ({:.1E}, {:.1E})'.format(
                                                    self.best_score, self.m.min(), self.m.max(), 
                                                    self.v.min(), self.v.max()))

                    self.mStopArr.append(self.m.max())
                    self.vStopArr.append(self.v.max())
                    self.optStopArr.append(self.best_score)
                    self.goodweights.append(self.weights)

                    if self.logModel_config is not None and i != 0 and (i + 1) % self.freqLog == 0:

                        outputFolder    = self.logModel_config['outputFolder']
                        on_Var          = self.logModel_config['on_Var']
                        preTrainedModel = self.logModel_config['preTrainedModel']
                        filename        = preTrainedModel.split('.')[0] + '_{:05d}'.format(i + 1) + '.' + preTrainedModel.split('.')[1]

                        with h5py.File( os.path.join(outputFolder, filename) ) as h5py_file:
                            h5py_file.create_dataset('weights', data=self.weights,     dtype=np.float64)
                            h5py_file.create_dataset('score',   data=self.best_score,  dtype=np.float64)
                            h5py_file.create_dataset('on_Var',  data=on_Var)

                        pbar.write('saving checkpoint.. {}'.format(os.path.join(outputFolder, filename)))

                    if i <= self.nStop:
                        continue

                    if (np.abs(np.mean(self.mStopArr)) < self.mStop) and \
                       (np.abs(np.mean(self.vStopArr)) < self.vStop) and \
                       (np.abs(np.mean(self.optStopArr) - self.best_score) < self.optStop) :
                       print('Early stopping here ...')
                       break

                except (KeyboardInterrupt, SystemExit):
                    print('Trying to stop the program ...')
                    return self.weights
                except Exception as e:
                    print(f'Error: Some problem with the step function: {e}')
                    return

            bestweightIndex   = np.nanargmin(self.optStopArr)

            self.best_score   = self.optStopArr[bestweightIndex]
            self.best_weights = self.goodweights[bestweightIndex]

            return self.best_score, self.best_weights, (self.iter_count, self.iter_score, self.weightsLog)

        except Exception as e:

            print('Unable to optimize \n{}'.format(str(e)))
            