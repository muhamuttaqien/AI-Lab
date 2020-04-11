import numpy as np
import pandas as pd

file_name = './simulator/data_fuzzy.xlsx'


class ExcelNet(object):
            
    def __init__(self, net, directory=file_name, time_step=24, total_agent=None):
            
        self.directory = directory
        self.net = net
        self.data = {}
        self.time_step = time_step
        self.total_agent = total_agent
            
        if self.exist():
            for s in self.sheets():

                self.data[s] = self.get_sheet(s)
                column_data = self.data[s].columns
                    
                if s == 'load':
                    self.load_net(s)
                if s == 'generator':
                    self.generator_net(s)
                if s == 'fuzzySet':
                    self.fuzzy_net(s)
                    
    def load_net(self, load):
        
        col = self.get_sheet(load).columns
        
        if len(col[1]) > 4:
            unit = col[1][4::]
        else:
            unit = '[MW]'
        
        data = self.get_sheet(load)
        for i in range(len(data)):
            self.net.create_load(data.values[i, 1], unit)
            
    def fuzzy_net(self, load):
        
        data = self.get_sheet(load).values
        for i in range(len(data)):
            self.net.create_fuzzy(data[i, 3], data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8], data[i, 9])

    def generator_net(self, gen):
        
        col = self.get_sheet(gen).columns
        col = col.values
        datas = self.get_sheet(gen)
        data = datas.values
        
        for i in range(len(data)):
            
            self.net.create_generator(data[i, 1], data[i, 2], data[i, 3], data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8], data[i, 9], data[i, 10], data[i, 11])
        
    def __getitem__(self, k):
        return self.data[k]
    
    def __setitem__(self, k, v):
        self.data[k] = v
        
    def get_sheet(self, name='', fail_accept=True):
        
        if fail_accept and name not in self.sheets():
            return None
        else:
            return pd.read_excel(self.directory, sheet_name=name)
        
    def sheets(self):
        
        return pd.ExcelFile(self.directory).sheet_names
    
    def read_excel(self, f):
        
        return pd.read_excel(f)
    
    def read_sheet(self, f):
        
        assert f in self.sheets(), 'named sheet is not in excel'
        return pd.read_excel(self.directory, sheet_name=f)
    
    def exist(self):
        
        try:
            _ = open(self.directory)
            return True
        except FileNotFoundError:
            return False
    
    def save(self):
        
        with pd.ExcelWriter(self.directory, engine='xlsxwriter') as writer:
            for k, v in self.data.items():
                v.to_excel(writer, sheet_name=k)
        