from __future__ import print_function, division
import numpy as np

class sbp_1d(object):
    def __init__(self, nx, dx, bc_type, order=2):
        self.nx = nx
        self.dx = dx
        self.bc_type = bc_type
        self.order = order

    def diff1_mat(self):
        '''
        return first dirivative matrix
        '''
        D1 = np.zeros((self.nx, self.nx))

        if self.order == 2: 
            d1 = np.diag(1/2 * np.ones(self.nx-1), +1)
            d2 = np.diag(-1/2 * np.ones(self.nx-1), -1)
        
        
            D1 = D1 + d1 + d2
        
            D1[0, 0] =-1
            D1[0, 1] = 1
        
            D1[-1, -2] = -1
            D1[-1, -1] = 1

        elif self.order == 4:
            d1 = np.diag(-1/12 * np.ones(self.nx-2), +2)
            d2 = np.diag(2/3 * np.ones(self.nx-1), +1)
            d3 = np.diag(-2/3 * np.ones(self.nx-1), -1)
            d4 = np.diag(1/12 * np.ones(self.nx-2), -2)
            
            D1 += d1 + d2 + d3 + d4

            a1 = np.array([-24.0/17.0, 59.0/34.0, -4.0/17.0, -3.0/34.0, 0.0, 0.0])
            a2 = np.array([-0.5, 0, 0.5, .00, 0.0, 0.0])
            a3 = np.array([4.0/43.0, -59.0/86.0, 0, 59.0/86.0, -4.0/43.0, 0])
            a4 = np.array([3.0/98.0, 0.0, -59/98, 0, 32.0/49.0, -4.0/49.0])
             
            D1[0, :6] = a1
            D1[1, :6] = a2
            D1[2, :6] = a3
            D1[3, :6] = a4

            D1[-1, -6:] = -a1[::-1]
            D1[-2, -6:] = -a2[::-1]
            D1[-3, -6:] = -a3[::-1]
            D1[-4, -6:] = -a4[::-1]
    
        self.D1 = D1/self.dx
        return self.D1
    
    def diff2_mat(self):
        '''
        return second dirivative matrix
        '''
        if self.order == 2: 
            D2 = np.diag(-2 * np.ones(self.nx))
        
            d1 = np.diag(np.ones(self.nx-1), +1)
            d2 = np.diag(np.ones(self.nx-1), -1)
        
            D2 = D2 + d1 + d2
        
            D2[0, 0] = 1
            D2[0, 1] = -2
            D2[0, 2] = 1 
        
            D2[-1, -1] = 1
            D2[-1, -2] = -2
            D2[-1, -3] = 1
             
        self.D2 = D2/(self.dx**2)
        return self.D2
   
    def S_mat(self):
        S = np.zeros((self.nx, self.nx))

        if self.order == 2:     
            #fully compatibal
            #S[0, :] = self.D1[0,:]
            #S[-1, :] = self.D1[-1,:]

            S[0,0] = -3/2
            S[0,1] = 2
            S[0,2] = -1/2
            
            S[-1,-1] = 3/2
            S[-1,-2] = -2
            S[-1,-3] = 1/2

        elif self.order == 4:
            #fully compatibal
            #S[0, :] = self.D1[0,:]
            #S[-1, :] = self.D1[-1,:]

            a = np.array([-11.0/6.0, 3.0, -3.0/2.0, 1.0/3.0])
            S[0, :4] = a 
            S[-1, -4:] = -a[::-1]
    
        self.S = S / self.dx
        return self.S
    
    def int_mat(self):
        
        if self.order == 2:
            H = np.eye(self.nx)
            H[0, 0] = 1/2
            H[-1, -1] = 1/2
        elif self.order == 4:
            vec = np.array([17.0/48.0, 59.0/48.0, 43.0/48.0, 49.0/48.0])
            vec1 = np.hstack((vec, np.ones(self.nx - 2*len(vec)), vec[::-1]))
            H = np.diag(vec1)
    
        self.H = H * self.dx
        return self.H
    
    def inv_int_mat(self):
        H = np.eye(self.nx)
            
        if self.order == 2: 
            H[0, 0] = 2
            H[-1, -1] = 2

        elif self.order == 4:
            vec = 1.0/np.array([17.0/48.0, 59.0/48.0, 43.0/48.0, 49.0/48.0])
            vec1 = np.hstack((vec, np.ones(self.nx - 2*len(vec)), vec[::-1]))
            H = np.diag(vec1)
    
        self.inv_H = H/self.dx
        return self.inv_H

    def B_mat(self):
        B = np.zeros((self.nx, self.nx))
        B[0, 0] = -1
        B[-1, -1] = 1
        self.B = B
        return self.B

    def BS_mat(self):
        self.S_mat()
        self.B_mat()
        self.BS = np.dot(self.B, self.S)
        return self.BS

    def d1_vec(self):
        if "B" not in self.__dict__:
            self.B_mat()
        self.d1_0 = self.S[0, :]
        self.d1_n = self.S[-1, :]
        return self.d1_0, self.d1_n
    
    def D3_mat(self):
        if self.order == 4:
            self.D3 = np.zeros((self.nx, self.nx))
            
            d1 = np.diag(1 * np.ones(self.nx-2), +2)
            d2 = np.diag(-3 * np.ones(self.nx-1), +1)
            d3 = np.diag(3 * np.ones(self.nx), 0)
            d4 = np.diag(-1 * np.ones(self.nx-1), -1)

            self.D3 += d1 + d2 + d3 + d4 
            self.D3[0,:4] = np.array([-1, 3, -3, 1])
            self.D3[-1,-4:] = np.array([-1, 3, -3, 1])
            self.D3[-2,-4:] = np.array([-1, 3, -3, 1])

            vec = np.array([-185893.0/301051.0, 79000249461.0/54642863857.0, -33235054191.0/54642863857.0,
                   -36887526683.0/54642863857.0, 26183621850.0/54642863857.0, -4386.0/181507.0])
            
            self.D3[2, :] *= 0
            self.D3[2, :len(vec)] = vec

            self.D3[-3, :] *= 0
            self.D3[-3, -len(vec):] = -vec[::-1]

    def D4_mat(self):
        if self.order == 4:
            self.D4 = np.zeros((self.nx, self.nx))
            
            d1 = np.diag(1 * np.ones(self.nx-2), +2)
            d2 = np.diag(-4 * np.ones(self.nx-1), +1)
            d3 = np.diag(6 * np.ones(self.nx), 0)
            d4 = np.diag(-4 * np.ones(self.nx-1), -1)
            d5 = np.diag(1 * np.ones(self.nx-2), -2)

            self.D4 += d1 + d2 + d3 + d4 + d5
            self.D4[0, :5] = np.array([1, -4, 6, -4, 1])
            self.D4[1, :5] = np.array([1, -4, 6, -4, 1])
            self.D4[-1, -5:] = np.array([1, -4, 6, -4, 1])
            self.D4[-2, -5:] = np.array([1, -4, 6, -4, 1])

    def C3_mat(self):
        if self.order == 4:
            vec1 = np.array([0, 0, 163928591571.0/53268010936.0, 189284.0/185893.0])
            vec2 = np.array([189284.0/185893.0, 0, 163928591571.0/53268010936.0, 0.0, 0.0])
            vec = np.hstack((vec1, np.ones(self.nx-len(vec1)-len(vec2)), vec2))

            self.C3 = np.diag(vec)

    def C4_mat(self):
        if self.order == 4:
            vec1 = np.array([0, 0, 1644330.0/301051.0, 156114.0/181507.0])
            vec = np.hstack((vec1, np.ones(self.nx - 2*len(vec1)), vec1[::-1]))
            self.C4 = np.diag(vec)

    def Ddx_mat(self, x):
        if "D1" not in self.__dict__:
            self.diff1_mat()
        self.Ddx = np.linalg.inv(np.diag(np.dot(self.D1, x)))
        return self.Ddx
        

class sbp_VC(sbp_1d):
    '''
    Variable Coefficient
    '''

    
    def R_VC_mat(self, k):
        k = np.dot(self.Ddx, k)
        if self.order == 2:
            if "D2" not in self.__dict__:
                super(sbp_VC, self).diff2_mat()
            C2 = np.eye(self.nx)
            C2[0, 0] = 0
            C2[-1, -1] = 0
            self.R_VC = np.dot(np.dot(np.dot((self.D2).T, C2), k), self.D2) * (1/4 * (self.dx)**3)
        elif self.order == 4:
            U = np.array([[48.0/17.0*(0.12e2/0.17e2 * k[0] + 0.59e2/0.192e3 * k[1] + 0.27010400129e11/0.345067064608e12 * k[2] + 0.69462376031e11/0.2070402387648e13 * k[3]), 
                 48.0/17.0*(-0.59e2/0.68e2 * k[0] - 0.6025413881e10/0.21126554976e11 * k[2] - 0.537416663e9/0.7042184992e10 * k[3]),
                 48.0/17.0*(0.2e1/0.17e2 * k[0] - 0.59e2/0.192e3 * k[1] + 0.213318005e9/0.16049630912e11 * k[3] + 0.2083938599e10/0.8024815456e10 * k[2]),
                 48.0/17.0*(0.3e1/0.68e2 * k[0] - 0.1244724001e10/0.21126554976e11 * k[2] + 0.752806667e9/0.21126554976e11 * k[3]),
                 48.0/17.0*(0.49579087e8/0.10149031312e11 * k[2] - 0.49579087e8/0.10149031312e11 * k[3]),
                 48.0/17.0*(-k[3]/0.784e3 + k[2]/0.784e3)],
                [48.0/59.0*(-0.59e2/0.68e2 * k[0] - 0.6025413881e10/0.21126554976e11 * k[2] - 0.537416663e9/0.7042184992e10 * k[3]),
                 48.0/59.0*(0.3481e4/0.3264e4 * k[0] + 0.9258282831623875e16/0.7669235228057664e16 * k[2] + 0.236024329996203e15/0.1278205871342944e16 * k[3]),
                 48.0/59.0*(-0.59e2/0.408e3 * k[0] - 0.29294615794607e14/0.29725717938208e14 * k[2] - 0.2944673881023e13/0.29725717938208e14 * k[3]),
                 48.0/59.0*(-0.59e2/0.1088e4 * k[0] + 0.260297319232891e15/0.2556411742685888e16 * k[2] - 0.60834186813841e14/0.1278205871342944e16 * k[3]),
                 48.0/59.0*(-0.1328188692663e13/0.37594290333616e14 * k[2] + 0.1328188692663e13/0.37594290333616e14 * k[3]),
                 48.0/59.0*(-0.8673e4/0.2904112e7 * k[2] + 0.8673e4/0.2904112e7 * k[3])],
                [48.0/43.0*(0.2e1/0.17e2 * k[0] - 0.59e2/0.192e3 * k[1] + 0.213318005e9/0.16049630912e11 * k[3] + 0.2083938599e10/0.8024815456e10 * k[2]),
                 48.0/43.0*(-0.59e2/0.408e3 * k[0] - 0.29294615794607e14/0.29725717938208e14 * k[2] - 0.2944673881023e13/0.29725717938208e14 * k[3]),
                 48.0/43.0*(k[0]/0.51e2 + 0.59e2/0.192e3 * k[1] + 0.13777050223300597e17/0.26218083221499456e17 * k[3] + 0.564461e6/0.13384296e8 * k[4] + 0.378288882302546512209e21/0.270764341349677687456e21 * k[2]),
                 48.0/43.0*(k[0]/0.136e3 - 0.125059e6/0.743572e6 * k[4] - 0.4836340090442187227e19/0.5525802884687299744e19 * k[2] - 0.17220493277981e14/0.89177153814624e14 * k[3]),
                 48.0/43.0*(-0.10532412077335e14/0.42840005263888e14 * k[3] + 0.1613976761032884305e19/0.7963657098519931984e19 * k[2] + 0.564461e6/0.4461432e7 * k[4]),
                 48.0/43.0*(-0.960119e6/0.1280713392e10 * k[3] - 0.3391e4/0.6692148e7 * k[4] + 0.33235054191e11/0.26452850508784e14 * k[2])],
                [48.0/49.0*(0.3e1/0.68e2 * k[0] - 0.1244724001e10/0.21126554976e11 * k[2] + 0.752806667e9/0.21126554976e11 * k[3]),
                 48.0/49.0*(-0.59e2/0.1088e4 * k[0] + 0.260297319232891e15/0.2556411742685888e16 * k[2] - 0.60834186813841e14/0.1278205871342944e16 * k[3]),
                 48.0/49.0*(k[0]/0.136e3 - 0.125059e6/0.743572e6 * k[4] - 0.4836340090442187227e19/0.5525802884687299744e19 * k[2] - 0.17220493277981e14/0.89177153814624e14 * k[3]),
                 48.0/49.0*(0.3e1/0.1088e4 * k[0] + 0.507284006600757858213e21/0.475219048083107777984e21 * k[2] + 0.1869103e7/0.2230716e7 * k[4] + k[5]/0.24e2 + 0.1950062198436997e16/0.3834617614028832e16 * k[3]),
                 48.0/49.0*(-0.4959271814984644613e19/0.20965546238960637264e20 * k[2] - k[5]/0.6e1 - 0.15998714909649e14/0.37594290333616e14 * k[3] - 0.375177e6/0.743572e6 * k[4]),
                 48.0/49.0*(-0.368395e6/0.2230716e7 * k[4] + 0.752806667e9/0.539854092016e12 * k[2] + 0.1063649e7/0.8712336e7 * k[3] + k[5]/0.8e1)],
                [0.49579087e8/0.10149031312e11 * k[2] - 0.49579087e8/0.10149031312e11 * k[3],
                 -0.1328188692663e13/0.37594290333616e14 * k[2] + 0.1328188692663e13/0.37594290333616e14 * k[3],
                 -0.10532412077335e14/0.42840005263888e14 * k[3] + 0.1613976761032884305e19/0.7963657098519931984e19 * k[2] + 0.564461e6/0.4461432e7 * k[4],
                 -0.4959271814984644613e19/0.20965546238960637264e20 * k[2] - k[5]/0.6e1 - 0.15998714909649e14/0.37594290333616e14 * k[3] - 0.375177e6/0.743572e6 * k[4],
                 0.8386761355510099813e19/0.128413970713633903242e21 * k[2] + 0.2224717261773437e16/0.2763180339520776e16 * k[3] + 0.5e1/0.6e1 * k[5] + k[6]/0.24e2 + 0.280535e6/0.371786e6 * k[4],
                 -0.35039615e8/0.213452232e9 * k[3] - k[6]/0.6e1 - 0.13091810925e11/0.13226425254392e14 * k[2] - 0.1118749e7/0.2230716e7 * k[4] - k[5]/0.2e1],
                [-k[3]/0.784e3 + k[2]/0.784e3 -0.8673e4/0.2904112e7 * k[2] + 0.8673e4/0.2904112e7 * k[3],
                 -0.960119e6/0.1280713392e10 * k[3],
                 -0.3391e4/0.6692148e7 * k[4] + 0.33235054191e11/0.26452850508784e14 * k[2],
                 -0.368395e6/0.2230716e7 * k[4] + 0.752806667e9/0.539854092016e12 * k[2] + 0.1063649e7/0.8712336e7 * k[3] + k[5]/0.8e1,
                 -0.35039615e8/0.213452232e9 * k[3] - k[6]/0.6e1 - 0.13091810925e11/0.13226425254392e14 * k[2] - 0.1118749e7/0.2230716e7 * k[4] - k[5]/0.2e1,
                 0.3290636e7/0.80044587e8 * k[3] + 0.5580181e7/0.6692148e7 * k[4] + 0.5e1/0.6e1 * k[6] + k[7]/0.24e2 + 0.660204843e9/0.13226425254392e14 * k[2] + 0.3e1/0.4e1 * k[5]]])
    
            L = np.array([[k[-8]/0.24e2 + 0.5e1/0.6e1 * k[-7] + 0.5580181e7/0.6692148e7 * k[-5] + 0.4887707739997e13/0.119037827289528e15 * k[-4] + 0.3e1/0.4e1 * k[-6] + 0.660204843e9/0.13226425254392e14 * k[-3] + 0.660204843e9/0.13226425254392e14 * k[-2],
                 -k[-7]/0.6e1 - 0.1618585929605e13/0.9919818940794e13 * k[-4] - k[-6]/0.2e1 - 0.1118749e7/0.2230716e7 * k[-5] - 0.13091810925e11/0.13226425254392e14 * k[-3] - 0.13091810925e11/0.13226425254392e14 * k[-2],
                 -0.368395e6/0.2230716e7 * k[-5] + k[-6]/0.8e1 + 0.48866620889e11/0.404890569012e12 * k[-4] + 0.752806667e9/0.539854092016e12 * k[-3] + 0.752806667e9/0.539854092016e12 * k[-2],
                 -0.3391e4/0.6692148e7 * k[-5] - 0.238797444493e12/0.119037827289528e15 * k[-4] + 0.33235054191e11/0.26452850508784e14 * k[-3] + 0.33235054191e11/0.26452850508784e14 * k[-2],
                 -0.8673e4/0.2904112e7 * k[-3] - 0.8673e4/0.2904112e7 * k[-2] + 0.8673e4/0.1452056e7 * k[-4],
                 -k[-4]/0.392e3 + k[-3]/0.784e3 + k[-2]/0.784e3],
                [-k[-7]/0.6e1 - 0.1618585929605e13/0.9919818940794e13 * k[-4] - k[-6]/0.2e1 - 0.1118749e7/0.2230716e7 * k[-5] - 0.13091810925e11/0.13226425254392e14 * k[-3] - 0.13091810925e11/0.13226425254392e14 * k[-2],
                 k[-7]/0.24e2 + 0.5e1/0.6e1 * k[-6] + 0.3896014498639e13/0.4959909470397e13 * k[-4] + 0.8386761355510099813e19/0.128413970713633903242e21 * k[-3] + 0.280535e6/0.371786e6 * k[-5] + 0.3360696339136261875e19/0.171218627618178537656e21 * k[-2],
                 -k[-6]/0.6e1 - 0.4959271814984644613e19/0.20965546238960637264e20 * k[-3] - 0.375177e6/0.743572e6 * k[-5] - 0.13425842714e11/0.33740880751e11 * k[-4] - 0.193247108773400725e18/0.6988515412986879088e19 * k[-2],
                 -0.365281640980e12/0.1653303156799e13 * k[-4] + 0.564461e6/0.4461432e7 * k[-5] + 0.1613976761032884305e19/0.7963657098519931984e19 * k[-3] - 0.198407225513315475e18/0.7963657098519931984e19 * k[-2],
                 -0.1328188692663e13/0.37594290333616e14 * k[-3] + 0.2226377963775e13/0.37594290333616e14 * k[-2] - 0.8673e4/0.363014e6 * k[-4],
                 k[-4]/0.49e2 + 0.49579087e8/0.10149031312e11 * k[-3] - 0.256702175e9/0.10149031312e11 * k[-2]],
                [48.0/49.0*(-0.368395e6/0.2230716e7 * k[-5] + k[-6]/0.8e1 + 0.48866620889e11/0.404890569012e12 * k[-4] + 0.752806667e9/0.539854092016e12 * k[-3] + 0.752806667e9/0.539854092016e12 * k[-2]),
                 48.0/49.0*(-k[-6]/0.6e1 - 0.4959271814984644613e19/0.20965546238960637264e20 * k[-3] - 0.375177e6/0.743572e6 * k[-5] - 0.13425842714e11/0.33740880751e11 * k[-4] - 0.193247108773400725e18/0.6988515412986879088e19 * k[-2]),
                 48.0/49.0*(k[-6]/0.24e2 + 0.1869103e7/0.2230716e7 * k[-5] + 0.507284006600757858213e21/0.475219048083107777984e21 * k[-3] + 0.3e1/0.1088e4 * k[-1] + 0.31688435395e11/0.67481761502e11 * k[-4] + 0.27769176016102795561e20/0.712828572124661666976e21 * k[-2]),
                 48.0/49.0*(-0.125059e6/0.743572e6 * k[-5] + k[-1]/0.136e3 - 0.23099342648e11/0.101222642253e12 * k[-4] - 0.4836340090442187227e19/0.5525802884687299744e19 * k[-3] + 0.193950157930938693e18/0.5525802884687299744e19 * k[-2]),
                 48.0/49.0*(0.260297319232891e15/0.2556411742685888e16 * k[-3] - 0.59e2/0.1088e4 * k[-1] - 0.106641839640553e15/0.1278205871342944e16 * k[-2] + 0.26019e5/0.726028e6 * k[-4]),
                 48.0/49.0*(-0.1244724001e10/0.21126554976e11 * k[-3] + 0.3e1/0.68e2 * k[-1] + 0.752806667e9/0.21126554976e11 * k[-2])],
                [48.0/43.0*(-0.3391e4/0.6692148e7 * k[-5] - 0.238797444493e12/0.119037827289528e15 * k[-4] + 0.33235054191e11/0.26452850508784e14 * k[-3] + 0.33235054191e11/0.26452850508784e14 * k[-2]),
                 48.0/43.0*(-0.365281640980e12/0.1653303156799e13 * k[-4] + 0.564461e6/0.4461432e7 * k[-5] + 0.1613976761032884305e19/0.7963657098519931984e19 * k[-3] - 0.198407225513315475e18/0.7963657098519931984e19 * k[-2]),
                 48.0/43.0*(-0.125059e6/0.743572e6 * k[-5] + k[-1]/0.136e3 - 0.23099342648e11/0.101222642253e12 * k[-4] - 0.4836340090442187227e19/0.5525802884687299744e19 * k[-3] + 0.193950157930938693e18/0.5525802884687299744e19 * k[-2]),
                 48.0/43.0*(0.564461e6/0.13384296e8 * k[-5] + 0.470299699916357e15/0.952302618316224e15 * k[-4] + 0.550597048646198778781e21/0.1624586048098066124736e22 * k[-2] + k[-1]/0.51e2 + 0.378288882302546512209e21/0.270764341349677687456e21 * k[-3]),
                 48.0/43.0*(-0.59e2/0.408e3 * k[-1] - 0.29294615794607e14/0.29725717938208e14 * k[-3] - 0.2234477713167e13/0.29725717938208e14 * k[-2] - 0.8673e4/0.363014e6 * k[-4]),
                 48.0/43.0*(-0.59e2/0.3136e4 * k[-4] - 0.13249937023e11/0.48148892736e11 * k[-2] + 0.2e1/0.17e2 * k[-1] + 0.2083938599e10/0.8024815456e10 * k[-3])],
                [48.0/59.0*(-0.8673e4/0.2904112e7 * k[-3] - 0.8673e4/0.2904112e7 * k[-2] + 0.8673e4/0.1452056e7 * k[-4]),
                 48.0/59.0*(-0.1328188692663e13/0.37594290333616e14 * k[-3] + 0.2226377963775e13/0.37594290333616e14 * k[-2] - 0.8673e4/0.363014e6 * k[-4]),
                 48.0/59.0*(0.260297319232891e15/0.2556411742685888e16 * k[-3] - 0.59e2/0.1088e4 * k[-1] - 0.106641839640553e15/0.1278205871342944e16 * k[-2] + 0.26019e5/0.726028e6 * k[-4]),
                 48.0/59.0*(-0.59e2/0.408e3 * k[-1] - 0.29294615794607e14/0.29725717938208e14 * k[-3] - 0.2234477713167e13/0.29725717938208e14 * k[-2] - 0.8673e4/0.363014e6 * k[-4]),
                 48.0/59.0*(0.9258282831623875e16/0.7669235228057664e16 * k[-3] + 0.3481e4/0.3264e4 * k[-1] + 0.228389721191751e15/0.1278205871342944e16 * k[-2] + 0.8673e4/0.1452056e7 * k[-4]),
                 48.0/59.0*(-0.6025413881e10/0.21126554976e11 * k[-3] - 0.59e2/0.68e2 * k[-1] - 0.537416663e9/0.7042184992e10 * k[-2])],
                [48.0/17.0*(-k[-4]/0.392e3 + k[-3]/0.784e3 + k[-2]/0.784e3),
                 48.0/17.0*(k[-4]/0.49e2 + 0.49579087e8/0.10149031312e11 * k[-3] - 0.256702175e9/0.10149031312e11 * k[-2]),
                 48.0/17.0*(-0.1244724001e10/0.21126554976e11 * k[-3] + 0.3e1/0.68e2 * k[-1] + 0.752806667e9/0.21126554976e11 * k[-2]),
                 48.0/17.0*(-0.59e2/0.3136e4 * k[-4] - 0.13249937023e11/0.48148892736e11 * k[-2] + 0.2e1/0.17e2 * k[-1] + 0.2083938599e10/0.8024815456e10 * k[-3]),
                 48.0/17.0*(-0.6025413881e10/0.21126554976e11 * k[-3] - 0.59e2/0.68e2 * k[-1] - 0.537416663e9/0.7042184992e10 * k[-2]),
                 48.0/17.0*(0.3e1/0.3136e4 * k[-4] + 0.27010400129e11/0.345067064608e12 * k[-3] + 0.234566387291e12/0.690134129216e12 * k[-2] + 0.12e2/0.17e2 * k[-1])]])
            
            nx_U, ny_U = U.shape
            R_VC = np.zeros([self.nx, self.nx])
            R_VC[:4, :ny_U] = U[:4,:]
            R_VC[4, :ny_U] = U[4,:]
            R_VC[4, ny_U] = -k[5]/0.6e1 + k[4]/0.8e1 + k[6]/0.8e1
            R_VC[5, :ny_U] = U[5,:]
            R_VC[5, ny_U:ny_U+2] = [-k[4]/0.6e1 - k[7]/0.6e1 - k[5]/0.2e1 - k[6]/0.2e1, -k[6]/0.6e1 + k[5]/0.8e1 + k[7]/0.8e1]
    
            nx_L, ny_L = L.shape
            R_VC[-6, -ny_L-2:-ny_L] = [-k[-7]/0.6e1 + k[-8]/0.8e1 + k[-6]/0.8e1, -k[-8]/0.6e1 - k[-5]/0.6e1 - k[-7]/0.2e1 - k[-6]/0.2e1]
            R_VC[-6, -ny_L:] = L[0,:]
            R_VC[-5, -ny_L-1] = -k[-6]/0.6e1 + k[-7]/0.8e1 + k[-5]/0.8e1
            R_VC[-5, -ny_L:] = L[1,:]
            R_VC[-4:, -ny_L:] = L[2:,:]
    
            for i in range(6, self.nx-6):
                R_VC[i, i-2:i+3] = [
                    -k[i-1]/0.6e1 + k[i-2]/0.8e1 + k[i]/0.8e1,
                    -k[i-2]/0.6e1 - k[i+1]/0.6e1 - k[i-1]/0.2e1 - k[i]/0.2e1,
                    k[i-2]/0.24e2 + 0.5e1/0.6e1 * k[i-1] + 0.5e1/0.6e1 * k[i+1] + k[i+2]/0.24e2 + 0.3e1/0.4e1 * k[i],
                    -k[i-1]/0.6e1 - k[i+2]/0.6e1 - k[i]/0.2e1 - k[i+1]/0.2e1,
                    -k[i+1]/0.6e1 + k[i]/0.8e1 + k[i+2]/0.8e1]
            

            #if "D3" not in self.__dict__:
            #    self.D3_mat()
            #if "C3" not in self.__dict__:
            #    self.C3_mat()
            #if "D4" not in self.__dict__:
            #    self.D4_mat()
            #if "C4" not in self.__dict__:
            #    self.C4_mat()

            ## B3 = k[i] + k[i+1]
            #idx = np.arange(self.nx)
            #idx = np.hstack((idx[1:], idx[-1]))
            #B3 = 0.5 * (k + np.diag(k[idx, idx]))

            ##self.R_VC = (1.0/18.0 /self.dx) * np.dot(np.dot(np.dot((self.D3).T, self.C3), B3), self.D3) \
            ##        + (1.0/144.0 /self.dx) * np.dot(np.dot(np.dot((self.D4).T, self.C4), k), self.D4)
            #self.R_VC =  np.dot(np.dot(np.dot((1.0/18.0 /self.dx) *(self.D3).T, self.C3), B3), self.D3) \
            #        +  np.dot(np.dot(np.dot((1.0/144.0 /self.dx) *(self.D4).T, self.C4), k), self.D4)
            self.R_VC = R_VC / self.dx**2
                    
        return self.R_VC
   
    def M_VC_mat(self, k):
        if "D1" not in self.__dict__:
            self.diff1_mat()
        if "R" not in self.__dict__:
            self.R_VC_mat(k)
        if "H" not in self.__dict__:
            self.int_mat()
        #self.M_VC = np.dot(np.dot(np.dot((self.D1).T, np.diag(k)), self.H), self.D1) + self.R_VC
        
        k = np.dot(self.Ddx, k)
        Mm1 = -0.5*(k[:-1] + k[1:])
        M0 = np.hstack([0.5*(k[0]+k[1]), 0.5*k[2:]+k[1:-1]+0.5*k[:-2], 0.5*(k[-1]+k[-2])])
        Mp1 = -0.5*(k[:-1] + k[1:])
        M_VC = np.diag(Mm1, +1) + np.diag(M0) + np.diag(Mp1, -1)
        self.M_VC = M_VC / self.dx

        return self.M_VC

    def BS_VC_mat(self, k):
        k = np.dot(self.Ddx, k)
        #super(sbp_VC, self).BS_mat()
        self.BS_mat()
        self.BS_VC = np.dot(np.diag(k), self.BS)
        return self.BS_VC

    def diff2_VC_mat(self, k):
        if "inv_H" not in self.__dict__:
            self.inv_int_mat()
        self.R_VC_mat(k)
        if self.order == 2:
            self.M_VC_mat(k)
            #self.BS_VC_mat(k) # updated in bc_mat
            self.D2_VC = np.dot(self.inv_H, (-self.M_VC + self.BS_VC))

        elif self.order == 4:
            D2_VC = -self.R_VC 
            k = np.dot(self.Ddx, k)
            D2_VC[0, :4] = -48.0/17.0*np.array([k[0,0]*(-11.0/6.0), k[0,0]*3, k[0,0]*(-3.0/2.0), k[0,0]*1.0/3.0])/self.dx**2 + D2_VC[0,:4]
            D2_VC[-1, -4:] = -48.0/17.0*np.array([k[-1,-1]*1.0/3.0, k[-1,-1]*(-3.0/2.0), k[-1,-1]*3.0, k[-1, -1]*(-11.0/6.0)])/self.dx**2 + D2_VC[-1, -4:]
            self.D2_VC = D2_VC
        return self.D2_VC

    def bc_mat(self, k):
        '''
        bc_type: 'dirichlet' or 'neumann'
            bc_type[0]: left boundary
            bc_type[1]: right boundary
        '''

        if "inv_H" not in self.__dict__:
            self.inv_int_mat()
        
        self.BS_VC_mat(k)

        E_0 = np.zeros((self.nx, self.nx))
        E_0[0, 0] = 1
        E_n = np.zeros((self.nx, self.nx))
        E_n[-1, -1] = 1
        e_0 = np.zeros(self.nx)
        e_0[0] = 1
        e_n = np.zeros(self.nx)
        e_n[-1] = 1

        if self.bc_type[0] == "dirichlet":
            #self.bcL = np.dot(self.Ddx, alphaD * (np.dot(np.dot(k, self.inv_H), e_0)) \
            #        + beta * np.dot(np.dot(self.inv_H, (self.BS_VC).T), e_0))
            #self.BCL = alphaD * (np.dot(np.dot(k, self.inv_H), E_0)) \
            #        + beta * np.dot(np.dot(self.inv_H, (self.BS_VC).T), E_0)
            self.bcL = np.dot(self.Ddx, np.dot(np.dot(self.inv_H, (self.BS_VC).T), e_0))
            self.BCL = -np.dot(np.dot(self.inv_H, (self.BS_VC).T), E_0)
        elif self.bc_type[0] == "neumann":
            self.bcL = np.dot(self.Ddx, np.dot(self.inv_H, e_0))  
            self.BCL = -np.dot(np.dot(self.inv_H, E_0), self.BS_VC)
        else:
            raise Exception("bc_type[0] (left boundary condition) \
                    should be 'dirichlet' or 'neumann'")
            
        if self.bc_type[1] == "dirichlet":
            #self.bcR = np.dot(self.Ddx, alphaD * (np.dot(np.dot(k, self.inv_H), e_n)) \
            #        + beta * np.dot(np.dot(self.inv_H, (self.BS_VC).T), e_n))
            #self.BCR = alphaD * (np.dot(np.dot(k, self.inv_H), E_n)) \
            #        + beta * np.dot(np.dot(self.inv_H, (self.BS_VC).T), E_n)
            self.bcR = np.dot(self.Ddx, np.dot(np.dot(self.inv_H, (self.BS_VC).T), e_n))
            self.BCR = -np.dot(np.dot(self.inv_H, (self.BS_VC).T), E_n)
        elif self.bc_type[1] == "neumann":
            self.bcR = np.dot(self.Ddx, np.dot(self.inv_H, e_n))
            self.BCR = -np.dot(np.dot(self.inv_H, E_n), self.BS_VC)
        else:
            raise Exception("bc_type[1] (right boundary condition) \
                    should be 'dirichlet' or 'neumann'")

        return self.bcL, self.bcR

    def D2bc_mat(self, k):
        '''
        D2 + L_sat + R_sat
        '''
        self.bc_mat(k)
        self.diff2_VC_mat(k)

        self.D2bc = np.dot(self.Ddx, self.D2_VC + self.BCL + self.BCR)

        return self.D2bc

    

class Coeff():
    def __init__(self, n0, beta0, k0, eta0, nx):
        self.n0 = n0
        self.n = np.ones(nx) * n0
        self.beta0 = beta0
        self.beta = np.ones(nx) * beta0
        self.k0 = k0
        self.k = np.ones(nx) * k0
        self.eta0 = eta0
        self.eta = np.ones(nx) * eta0
        self.nx = nx

        self.kappa = self.k / self.eta

    def update_n(self, p):
        """
        porosity
        """
        return self.n

    def update_beta(self, p):
        """
        compressibility
        """
        return self.beta
    
    def update_k(self, p, stage=1):
        """
        permeability
        """
        self.k = self.k0 * np.ones_like(p) * np.exp(-p/1e7)
        return self.k

    def update_eta(self, p):
        """
        viscosity 
        """
        return self.eta

    def update_kappa(self, p, stage=1):
        """
        kappa = k/eta
        """
        k = self.update_k(p)
        eta = self.update_eta(p)
        self.kappa = k/eta
        return self.kappa

class Solver():
    def implicit(self, SBP_VC, coeff, x0, g=None, min_diff = 1e-3, max_iter = 10):
        x = x0

        if SBP_VC.bc_type[0] == "neumann":
            coeff_L = coeff.q_L
        elif SBP_VC.bc_type[1] == "dirichlet":
            coeff_L = coeff.p_L
        if SBP_VC.bc_type[1] == "neumann":
            coeff_R = coeff.q_R
        elif SBP_VC.bc_type[1] == "dirichlet":
            coeff_R = coeff.p_R

        for i in range(max_iter):
            coeff.update_kappa(x);
            SBP_VC.bc_mat(coeff.kappa)
            SBP_VC.D2bc_mat(coeff.kappa)
            M = coeff.I - coeff.dt * np.dot(np.diag(1/(coeff.n * coeff.beta)), SBP_VC.D2bc) 
            #M = coeff.I - coeff.dt * 1/(coeff.n * coeff.beta) * SBP_VC.D2bc 

            if g is not None:
                b = coeff.dt * (np.dot(np.diag(1/(coeff.n * coeff.beta)), (coeff_R * SBP_VC.bcR - coeff_L * SBP_VC.bcL)) + g)
                #b = coeff.dt * (1/(coeff.n * coeff.beta) * (coeff.p_R * SBP_VC.bcR + coeff.p_L * SBP_VC.bcL) + g)
            else:
                b = coeff.dt * np.dot(np.diag(1/(coeff.n * coeff.beta)), (coeff_R * SBP_VC.bcR - coeff_L * SBP_VC.bcL))
                #b = coeff.dt * 1/(coeff.n * coeff.beta) * (coeff.p_R * SBP_VC.bcR + coeff.p_L * SBP_VC.bcL)

            tmp = x.copy()
            x = np.dot(np.linalg.inv(M), x0 + b)
            diff = np.max(np.abs(x - tmp))
            #print(diff)
            #if diff < min_diff:
            #    print('iter: ', i)
            #    return x
        #print('Maximum iteration reached!')
        return x


