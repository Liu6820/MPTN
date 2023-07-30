from model.tools import *
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from model.data_loader import *
from model.predict import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, f1_score, auc
import csv
import sys
import numpy as np

class Main(object):

	def __init__(self, params):
		self.p = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model()
		self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

	def load_data(self):
		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'valid','test']:
			for line in open('/root/GPKG/data/{}/{}.tsv'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.upper, line.strip().split('\t'))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim


		self.data	= ddict(list)
		sr2o		= ddict(set)

		for split in ['train', 'test', 'valid']:
			for line in open('/root/GPKG/data/{}/{}.tsv'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.upper, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

		self.triples = ddict(list)
		#print(sr2o)
		for (sub, rel), obj in self.sr2o.items():
			#print(sr2o[(sub, rel)])
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
			# for o in obj:
			# 	self.triples['train'].append({'triple':(sub, rel, o), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		# for split in ['train','test', 'valid']:
		# 	for sub, rel, obj in self.data[split]:
		# 		rel_inv = rel + self.p.num_rel
		# 		#if split=='train':self.triples[split].append({'triple':(sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        #     	self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
		# 		self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})
		# for split in ['train', 'test', 'valid']:
		# 	for sub, rel, obj in self.data[split]:
		# 		rel_inv = rel + self.p.num_rel
		# 		if split=='train':
		# 			self.triples[split].append({'triple':(sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
		# 		else:
		# 			self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
		# 			self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train'		:   get_data_loader(TrainDataset, 'train', 	self.p.batch_size),
			'valid_head'	:   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail'	:   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head'	:   get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail'	:   get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}

		self.chequer_perm	= self.get_chequer_perm()
		self.edge_index, self.edge_type = self.construct_adj()

	def construct_adj(self):
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		return edge_index, edge_type

	def get_chequer_perm(self):
		ent_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])
		rel_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])

		comb_idx = []
		for k in range(self.p.perm):
			temp = []
			ent_idx, rel_idx = 0, 0

			for i in range(self.p.k_h):
				for j in range(self.p.k_w):
					if k % 2 == 0:
						if i % 2 == 0:
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
						else:
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
					else:
						if i % 2 == 0:
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
						else:
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;

			comb_idx.append(temp)

		chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
		return chequer_perm


	def add_model(self):
		model = GPKG_PREDICT(self.edge_index, self.edge_type, self.chequer_perm, params=self.p)
		model.to(self.device)
		return model

	def read_batch(self, batch, split):
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		state				= torch.load(load_path)
		state_dict			= state['state_dict']
		self.best_val_mrr 		= state['best_val']['mrr']
		self.best_val 			= state['best_val']

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])
	# def ro_curve(self, y_pred, y_label, b):
	# 	'''
	# 		y_pred is a list of length n.  (0,1)
	# 		y_label is a list of same length. 0/1
	# 		https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
	# 	'''
	# 	y_label = np.array(y_label)
	# 	#y_label = y_label.data.cpu().numpy()
	# 	y_pred = np.array(y_pred)    
	# 	# f = np.zeros()
	# 	# t = np.zeros()
	# 	# _ = np.zeros()
	# 	r = 0
	# 	for i in range(b):
	# 		f1, t1, a = roc_curve(y_label[i], y_pred[i])
	# 		#print(f1.shape, t1.shape, a.shape)
	# 		r1 = auc(f1, t1)
	# 		#print(r1)
	# 		# f = f + f1
	# 		# t = t +t1
	# 		# _ = _ + a
	# 		r = r + r1
	# 	# f = f/b
	# 	# t = t/b
	# 	# _ = _/b
	# 	# r = r/b
		
	# 	return r
	def ro_curve(self, y_pred, y_label, b):
		y_label = np.array(y_label)
		y_pred = np.array(y_pred)
		#lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
#   plt.plot([0,1], [no_skill, no_skill], linestyle='--')
		a = 0
		for i in range(b):
			a1 = average_precision_score(y_label[i], y_pred[i])
			a = a + a1
		return a
	def evaluate(self, split, epoch=0):		
		z = 516
		left_results, r = self.predict(split=split, mode='tail_batch')
		#print(pred.shape)
		# f = f/249345
		# t = t/249345
		# c = c/249345
		r = r/z
		print(r)
		print(left_results['count'])
		right_results, r1 = self.predict(split=split, mode='head_batch')
		# f = f1/249345
		# t = t1/249345
		# c = d/249345
		r1 = r1/z
		print(r1)
		# self.fpr[0] = (f + f1)/2
		# self.tpr[0] = (t + t1)/2
		# self._[0] = (c + d)/2
		r = (r + r1)/2
		print(r)
	# 	lw = 2
	# 	plt.plot(self.fpr[0], self.tpr[0],
    #      	lw=lw, label= 'method_name' + ' (area = %0.2f)' % self.roc_auc[0])
	# 	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	# 	plt.xlim([0.0, 1.0])
	# 	plt.ylim([0.0, 1.05])
    # # plt.xticks(font="Times New Roman",size=18,weight="bold")
    # # plt.yticks(font="Times New Roman",size=18,weight="bold")
	# 	fontsize = 14
	# 	plt.xlabel('False Positive Rate', fontsize = fontsize)
	# 	plt.ylabel('True Positive Rate', fontsize = fontsize)
    # #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
	# 	plt.legend(loc="lower right")
	# 	plt.savefig('Ficture' + ".pdf")
	# 	#print(right_results['count'])
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}, hits@1 : {:.5}, hits@3 : {:.5}, hits@5 : {:.5}, hits@10 : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr'], results['hits@1'], results['hits@3'], results['hits@5'], results['hits@10']))
		return results
	# def evaluate(self, split, epoch=0):		
	# 	left_results  = self.predict(split=split, mode='tail_batch')
	# 	right_results = self.predict(split=split, mode='head_batch')
	# 	results       = get_combined_results(left_results, right_results)
	# 	self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}, hits@1 : {:.5}, hits@3 : {:.5}, hits@5 : {:.5}, hits@10 : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr'], results['hits@1'], results['hits@3'], results['hits@5'], results['hits@10']))
	# 	return results
	def predict(self, split='valid', mode='tail_batch'):
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
			#pred1 = torch.Tensor([[]])
			r1 = 0
			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel, None)
				r = self.ro_curve(pred, label, pred.shape[0])
				# f1 = f1 + f
				# t1 = t1 + t
				# c1 = c1 + c
				r1 = r1 + r
				#print(pred.shape[0])
				#print(step)
				
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				target_label = label[b_range, obj]
				print(target_label, target_label.shape)
				pred 			= torch.where(label.byte(), torch.zeros_like(pred), pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 1000 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

		return results, r1

	# def predict(self, split='valid', mode='tail_batch'):
	# 	self.model.eval()

	# 	with torch.no_grad():
	# 		results = {}
	# 		train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

	# 		for step, batch in enumerate(train_iter):
	# 			sub, rel, obj, label	= self.read_batch(batch, split)
	# 			# x_node, x, pred, obj_emb	= self.model.forward(sub, rel, obj, None)
	# 			pred	= self.model.forward(sub, rel, None)
	# 			b_range			= torch.arange(pred.size()[0], device=self.device)
	# 			target_pred		= pred[b_range, obj]
	# 			pred 			= torch.where(label.byte(), torch.zeros_like(pred), pred)
	# 			pred[b_range, obj] 	= target_pred
	# 			ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

	# 			ranks 			= ranks.float()
	# 			results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
	# 			results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
	# 			results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
	# 			for k in range(10):
	# 				results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

	# 			if step % 1000 == 0:
	# 				self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

	# 	return results

	def run_epoch(self, epoch):
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])

		for step, batch in enumerate(train_iter):
			#print(batch)
			self.optimizer.zero_grad()
			
			sub,rel,obj,label,neg_ent,sub_samp = self.read_batch(batch, 'train')
			#print(sub.shape)
			#print(label)
			#x1_node, x, pred, tail_emb1 = self.model.forward(sub, rel, obj, neg_ent)
			#pred = self.model.forward(sub, rel, obj, neg_ent)
			pred = self.model.forward(sub, rel, neg_ent)
			# tail_emb1 = F.normalize(tail_emb1, dim=1)
			# x1_node = F.normalize(x1_node, dim=1)
        
       		#  # calculate SupCon loss
			# features1 = torch.cat((x1_node.unsqueeze(1), tail_emb1.unsqueeze(1)), dim=1)
        	# # features2 = torch.cat((x2_node.unsqueeze(1), tail_emb1.unsqueeze(1)), dim=1)
        
        	# # SupCon Loss
			# supconloss1 = self.model.supconloss(features1, labels=obj, mask=None)
			# celoss	= self.model.loss(x, obj, sub_samp)
			# loss = (supconloss1) + (celoss)
			#loss  = self.model.loss(x, obj, sub_samp)
			#print(pred)
			loss  = self.model.loss(pred, label, sub_samp)
			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 1000 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}, \t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss

	def fit(self):
		self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0.
		val_mrr = 0
		#save_path = os.path.join('/root/GPKG/code/model_saved', self.p.name)
		#save_path = os.path.join('/root/GPKG/code/model_saved', 'sota_biokg')
		#save_path = os.path.join('/root/GPKG/code/model_saved', 'testrun_36616030')
		save_path = os.path.join('/root/GPKG/code/model_saved', 'testrun_fcaf0c24')
		#save_path = os.path.join('/root/GPKG/code/model_saved', 'testrun_fe74bc75')
		# if self.p.restore:
		# 	self.load_model(save_path)
		# 	self.logger.info('Successfully Loaded previous model')
		self.load_model(save_path)
		self.logger.info('Successfully Loaded previous model')
		for epoch in range(self.p.max_epochs):
			train_loss	= self.run_epoch(epoch)
			val_results	= self.evaluate('valid', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				self.best_val		= val_results
				self.best_val_mrr	= val_results['mrr']
				self.best_epoch		= epoch
				self.save_model(save_path)
			self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Valid MRR: {:.5}, \n\n\n'.format(epoch, train_loss, self.best_val_mrr))

		
		# Restoring model corresponding to the best validation performance and evaluation on test data
		self.logger.info('Loading best model, evaluating on test data')
		self.load_model(save_path)		
		self.evaluate('test')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Dataset and Experiment name
	#parser.add_argument('-data',           dest='dataset',         default='biokg',            		help='Dataset to use for the experiment')
	parser.add_argument('-data',           dest='dataset',         default='openbiolink',            		help='Dataset to use for the experiment')
	parser.add_argument('-name',            			default='testrun_'+str(uuid.uuid4())[:8],	help='Name of the experiment')

	# Training parameters
	parser.add_argument('-gpu',		type=str,               default='0',					help='GPU to use, set -1 for CPU')
	parser.add_argument('-neg_num',        dest='neg_num',         default=1000,    	type=int,       	help='Number of negative samples to use for loss calculation')
	parser.add_argument('-batch',          dest='batch_size',      default=128,    	type=int,       	help='Batch size')
	parser.add_argument('-l2',		type=float,             default=0.0,					help='L2 regularization')
	parser.add_argument('-lr',		type=float,             default=0.001,					help='Learning Rate')
	parser.add_argument('-epoch',		dest='max_epochs', 	default=0,		type=int,  		help='Maximum number of epochs')
	parser.add_argument('-num_workers',	type=int,               default=30,                      		help='Maximum number of workers used in DataLoader')
	parser.add_argument('-seed',           dest='seed',            default=41504,   		type=int,       	help='Seed to reproduce results')
	parser.add_argument('-restore',   	dest='restore',       	action='store_true',            		help='Restore from the previously saved model')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')
	parser.add_argument("--op", type=str, default="corr", help="aggregation opration")
	parser.add_argument("--beta", type=float, default=0, help="description for experiment")


	
	# Model parameters
	parser.add_argument('-lbl_smooth',     dest='lbl_smooth',	default=0.1,		type=float,		help='Label smoothing for true labels')
	parser.add_argument('-embed_dim',	type=int,              	default=200,                   		help='Embedding dimension for entity and relation, ignored if k_h and k_w are set')
	parser.add_argument('-num_bases',	dest='num_bases', 	default=-6,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
 
	parser.add_argument("--ent_dim", type=int, default=200, help="dimension of entities embeddings")
	parser.add_argument("--rel_dim", type=int, default=200, help="dimension of relations embeddings")
	parser.add_argument("--encoder_drop", type=float, default=0.1, help="dropout ratio for encoder")
 
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')

	parser.add_argument('-bias',      	dest='bias',          	action='store_true',            		help='Whether to use bias in the model')
	parser.add_argument('-form',		type=str,               default='plain',            			help='The reshaping form to use')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   		type=int, 		help='Width of the reshaped matrix')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   		type=int, 		help='Height of the reshaped matrix')
	parser.add_argument('-num_filt',  	dest='num_filt',      	default=200,     	type=int,       	help='Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz',        	default=7,     		type=int,       	help='Kernel size to use')
	parser.add_argument('-perm',      	dest='perm',          	default=1,      	type=int,       	help='Number of Feature rearrangement to use')
	parser.add_argument('-hid_drop',  	dest='hid_drop',      	default=0.3,    	type=float,     	help='Dropout for Hidden layer')
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop',     	default=0.3,    	type=float,     	help='Dropout for Feature')
	parser.add_argument('-inp_drop',  	dest='inp_drop',      	default=0.1,    	type=float,     	help='Dropout for Input layer')
	parser.add_argument("--temp1", type=float, default=0.07, help="temperature of contrastive loss")

	# Logging parameters
	parser.add_argument('-logdir',    	dest='log_dir',       	default='/root/GPKG/code/log/',               		help='Log directory')
	parser.add_argument('-config',    	dest='config_dir',    	default='/root/GPKG/code/config/',            		help='Config directory')
	
	# parser.add_argument('-logdir',    	dest='log_dir',       	default='./log/',               		help='Log directory')
	# parser.add_argument('-config',    	dest='config_dir',    	default='./config/',            		help='Config directory')
	

	args = parser.parse_args()
	
	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Main(args)
	model.fit()
