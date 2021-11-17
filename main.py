
from examples.tgcn_example import run_TGCN
from examples.a3tgcn_example import run_A3TGCN
from examples.agcrn_example import run_AGCRN
from examples.dcrnn_example import run_DCRNN
from examples.dygrencoder_example import run_DyGrEncoder
from examples.evolvegcnh_example import run_EvolveGCNH
from examples.evolvegcno_example import run_EvolveGCNO
from examples.gclstm_example import run_GCLSTM
from examples.lrgcn_example import run_LRGCN
from examples.mpnnlstm_example import run_MPNNLSTM
from examples.gconvgru_example import run_GConvGRU
from examples.gconvlstm_example import run_GConvLSTM

from loader import TrafficDataLoader
import networkx as nx
import csv

G_Los = nx.read_edgelist('./data/los_vis_edgelist.txt', nodetype=int, create_using= nx.DiGraph())
G_D4= nx.read_edgelist('./data/d4_vis_edgelist.txt', nodetype=int, create_using= nx.DiGraph())
G_D8= nx.read_edgelist('./data/d8_vis_edgelist.txt', nodetype=int, create_using= nx.DiGraph())


print( 'Los_Angeles', G_Los.number_of_nodes(), 'PeMSD4', G_D4.number_of_nodes(), 'PeMSD8',G_D8.number_of_nodes())

node_number={'Los_Angeles':G_Los.number_of_nodes(), 'PeMSD4':G_D4.number_of_nodes(), 'PeMSD8':G_D8.number_of_nodes()}




if __name__ == "__main__":

   for name in ['Los_Angeles', 'PeMSD8', 'PeMSD4']: 

      results= {}

      loader=TrafficDataLoader(data_name=name)


      cost_r, cost_a= run_TGCN(loader, name)
      results['TGCN']=[name, cost_r, cost_a]

      cost_r, cost_a= run_EvolveGCNH(loader, node_number[name],name)
      results['EvolveGCNH']=[name, cost_r, cost_a]

      cost_r, cost_a = run_EvolveGCNO(loader, name)
      results['EvolveGCNO'] = [name, cost_r, cost_a]

      cost_r, cost_a= run_A3TGCN(loader, name)
      results['A3TGCN']=[name, cost_r, cost_a]

      cost_r, cost_a= run_AGCRN(loader, node_number[name], name)
      results['AGCRN']=[name, cost_r, cost_a]

      cost_r, cost_a= run_DCRNN(loader, name)
      results['DCRNN']=[name, cost_r, cost_a]

      cost_r, cost_a= run_LRGCN(loader, name)
      results['LRGCN']=[name, cost_r, cost_a]

      cost_r, cost_a= run_DyGrEncoder(loader, name)
      results['DyGrEncoder']=[name, cost_r, cost_a]


      cost_r, cost_a= run_GCLSTM(loader, name)
      results['GCLSTM']=[name, cost_r, cost_a]

      cost_r, cost_a=run_MPNNLSTM(loader, node_number[name], name)
      results['MPNNLSTM']=[name, cost_r, cost_a]

      cost_r, cost_a=run_GConvGRU(loader, name)
      results['GConvGRU']=[name, cost_r, cost_a]

      cost_r, cost_a=run_GConvLSTM(loader, name)
      results['GConvLSTM']=[name, cost_r, cost_a]



      print(results)
      with open('./results/results_%s.csv'%name, 'w') as f:
         writer = csv.writer(f)
         header = ['model','dataset', 'RMSE', 'MAE']
         writer.writerow(header)


         for i in results:
           row=[]
           row.append(i)
           row.append(results[i][0])
           row.append(round(results[i][1],4))
           row.append(round(results[i][2],4))
           writer.writerow(row)






