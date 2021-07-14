import numpy as np
from sklearn.decomposition import PCA

class SOM(object):
    def __init__(self, x, y, epochs, vec_size, metric = 'Euclidean',lr0 = 0.01, sigma0 = 5, initialisation = 'RandomNormal'):
        """ Initialize the SOM object with a given map size
        
        :param x: {int} width of the map
        :param y: {int} height of the map
        :param lr0: {float} initial learning rate to start the training. Default 0.01
        :param epochs: {int} Number of epochs to tran the SOM
        :param vec_size: {int} Dimension of the SOM vectors
        :param metric: {str} Metric used to estimate ditances: chi2, Euclidean or Manhattan
        :param sigma0: {float} Inital smoothing sigma. Default 5
        :param initialisation: {String} RandomNormal or RandomSampling. RandomNormal initialises the map with random vectors from a normal distribution with mean and std from the data. RandomSampling selects randomly data samples as initial som vectors Default 'RandomNormal'
        """
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.sigma = sigma0
        self.lr = lr0
        self.lrs = None
        self.sigmas = None
        self.epochs =epochs
        self.epoch = 0
        self.vec_size = vec_size
        self.som_vector = np.random.normal(size=(self.x, self.y, self.vec_size))
        self.metric = metric
        self.samples_epoch = None
        self.initialisation = None
        

    def calculate_distances(self, data_obj):
        """ Estimates the distance from the datapoint to the SOM vector.
        :param data_obj: {numpy.ndarray} Current datapoint. If it also has errors, the first
        entries are the values and are followed by the errors. Errors are only needed if 'chi2'
        is used as metric."""
                
        if self.metric == 'chi2':
            d = (data_obj[:self.vec_size] - self.som_vector)**2 / data_obj[self.vec_size:]**2
        elif self.metric == 'Euclidean':
            d = (data_obj[:self.vec_size] - self.som_vector)**2
        elif self.metric == 'Manhattan':
            d = np.abs(data_obj[:self.vec_size] - self.som_vector)
            
        d = np.nansum(d,2)
        return d
    
    def winner_som_cell(self, distances):
        """ Estimates the SOM cell where the current datapoint is allocated
        :param distances: {numpy.ndarray} Distance from the datapoint to every SOM vector cell 
        :return r {tuple}: Indiices of the best matchiing cell"""
        
        d_min = distances.min()
        t_idx = np.unravel_index(distances.argmin(), distances.shape)
        t = self.som_vector[t_idx[0],t_idx[1],:]
        
        return t,t_idx
    
    def som_error_estimation(self,bmu_vec,data_obj):
        """ Estimates the error between the bmu vector and the datapoint
        :param bmu_vec: {numpy.ndarray} bmu vector    
        :param data_obj: {numpy.ndarray} datapoint """
        
        if self.metric == 'chi2':
            d = (data_obj[:self.vec_size] - bmu_vec)**2 / data_obj[self.vec_size:]**2
        elif self.metric == 'Euclidean':
            d = (data_obj[:self.vec_size] - bmu_vec)**2
        elif self.metric == 'Manhattan':
            d = np.abs(data_obj[:self.vec_size] - bmu_vec)
            
        d = np.nansum(d)
            
        return d
    
    def som_distance_grid(self,bmu_idx):
        dx = np.arange(0,self.x,1) - bmu_idx[0]
        dy = np.arange(0,self.y,1) - bmu_idx[1]
        ##Application of PBC:
        
        
        dx = np.where(dx> 0.5*self.x, dx - self.x, dx) 
        dx = np.where(dx<= -0.5*self.x, + self.x + dx, dx) 
                        
        dy = np.where(dy> 0.5*self.y, dy - self.y, dy) 
        dy = np.where(dy<= -0.5*self.y, +self.y + dy, dy) 

        grid = dx[:,None]**2 + dy[None,:] **2
        d = np.sqrt(grid)
        return d

                 

    def som_training_samp(self, data_obj):
        """ Perform one iteration over a data point and updates the SOM accordingly
        :param data_obj: {numpy.ndarray} Current datapoint. If it also has errors, the first
        entries are the values and are followed by the errors. Errors are only needed if 'chi2'
        is used as metric."""
                
        d = self.calculate_distances(data_obj)
        bmu, bmu_idx = self.winner_som_cell(d)

        # smooth the distances with the current sigma
        d_cells = self.som_distance_grid(bmu_idx)
        beta = np.nan_to_num(np.exp(-d_cells**2/(2*self.sigmas[self.epoch]**2)),0)
        beta = np.where(beta>3*self.sigma, 1000, beta) 
                
        # update SOM cell vector
        self.som_vector = self.som_vector - beta[:,:,None] * self.lr * np.nan_to_num(self.som_vector - data_obj[:self.vec_size])
        
        return bmu
        



    def train(self, data, samples_epoch=None):
        """ Train the SOM on the given data for several iterations
        :param data: {numpy.ndarray} data to train on
        :param samples_epoch: {int} number of data samples used to train in each epoch.
        """

        if self.initialisation == 'RandomNormal':
        
            self.som_vector = np.random.normal(data.mean(), data.std(),size=(self.x, self.y, self.vec_size))
        
        elif self.initialisation == 'RandomSampling':
            self.som_vector = np.random.normal(data.mean(), data.std(),size=(self.x, self.y, self.vec_size))
            for i in range(self.x):
                for j in range(self.y):
                    self.som_vector[i,j] = data[np.random.randint(0, len(data))]
            

        self.samples_epoch = samples_epoch#len(data) / 10
        epoch_list = np.arange(1, self.epochs+1, 1)
        
        
        self.lrs = np.linspace(self.lr,0.01,self.epochs)
        #self.sigmas = np.linspace(self.sigma,1,self.epochs)
        
        #self.lrs = self.lr**(epoch_list/self.epochs)
        self.sigmas = self.sigma * (1/self.sigma)**(epoch_list/self.epochs)

        
        for e in range(self.epochs):
            indx = np.random.choice(np.arange(len(data)), self.samples_epoch)
            error = 0
            for i in range(samples_epoch):
                bmu_somvec = self.som_training_samp(data[indx[i]])
                error += self.som_error_estimation(bmu_somvec,data[indx[i]])
                
                
            self.epoch = self.epoch +1
            if self.epoch < self.epochs:
                self.sigma = self.sigmas[self.epoch]
                self.lr = self.lrs[self.epoch]
                
            #print('epoch %s'%e, 'lr: %s'%self.lr, 'sigma: %s'%self.sigma, 'error: %s'%error)
                


    def test_obj(self, obj_vect):
        
        """ Test in which cell the objects are allocated
        :param obj_vect: :param data_obj: {numpy.ndarray} Current datapoint. If it also has errors, the first
        entries are the values and are followed by the errors. Errors are only needed if 'chi2'
        is used as metric.
        :returns bmu: {numpy.ndarray} Array with the best matching units (matching cells)
        """
        
        if self.metric == 'chi2':
            d = (obj_vect[:,None,None,:self.vec_size] - self.som_vector)**2 / obj_vect[:,None,None,self.vec_size:]**2
            d = np.nansum(d,3)
        elif self.metric == 'Euclidean':
            d = (obj_vect[:,None,None,:self.vec_size] - self.som_vector)**2
            d = np.nansum(d,3)
        elif self.metric == 'Manhattan':
            d = np.abs((obj_vect[:,None,None,:self.vec_size] - self.som_vector))
            d = np.nansum(d,3)        
    
            
        d = d.reshape(len(d),d.shape[1]*d.shape[2])
        bmu = d.argmin(1)
        
        return bmu

    
