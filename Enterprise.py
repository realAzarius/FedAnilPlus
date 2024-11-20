import numpy as np
import math
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DatasetLoad import DatasetLoad
from DatasetLoad import AddGaussianNoise
from sklearn.datasets import make_blobs
from torch import optim
import random
import copy
import time
from sys import getsizeof
from Crypto.PublicKey import RSA
from hashlib import sha256
from Models import ConcatModel, CombinedModel, Generator
# FedAnil+: Consortium Blockchain
from Block import Block
# FedAnil+: Consortium Blockchain
from Consortium_Blockchain import Consortium_Blockchain
from torchvision import transforms
# FedAnil+: Adaptive Huffman Codeing
from bitarray import bitarray
# FedAnil+: Adaptive Huffman Codeing AHC编码
from bitarray.util import huffman_code as adaptive_huffman
# FedAnil+: Affinity Propagation Ap聚合亲和传播
from sklearn.cluster import AffinityPropagation
import sys
import warnings

import tenseal as ts  # 用于加密计算库，适合同态加密

import torch.nn as nn

warnings.filterwarnings('ignore')
# FedAnil+: Cosine Similarity (Threshold 1 and 2) 余弦相似度的阈值
TRESHOLD1 = -0.7
TRESHOLD2 = 0.7

'''三种模型类型，卷积神经网络、Glove词向量、ResNet'''
m_type = ["cnn", "glove", "resnet"]
'''计数器'''
flcnt = 0
lastprc = 0


class Enterprise:
    def __init__(self, idx, assigned_train_ds, assigned_test_dl, local_batch_size, learning_rate, loss_func, opti,
                 network_stability, net, dev, miner_acception_wait_time, miner_accepted_transactions_size_limit,
                 validator_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed,
                 even_computation_power, is_malicious, noise_variance, check_signature, not_resync_chain,
                 malicious_updates_discount, knock_out_rounds, lazy_local_enterprise_knock_out_rounds):
        """
        :param idx: 节点的唯一标识符
        :param assigned_train_ds: 分配给节点的训练数据集
        :param assigned_test_dl: 分配给节点的测试数据加载器DataLoader
        :param local_batch_size: 本地批次大小
        :param learning_rate: 学习率
        :param loss_func: 损失函数
        :param opti: 优化器 SGD或者Adam
        :param network_stability: 网络稳定性
        :param net: 深度学习模型
        :param dev: 设备参数，比如CPU，GPU
        :param miner_acception_wait_time: 矿工接受交易的等待时间
        :param miner_accepted_transactions_size_limit: 每个矿工节点可以接受的交易数量上限
        :param validator_threshold:  验证者阈值，在共识算法中用于验证区块或交易的有效性
        :param pow_difficulty: 工作量证明（PoW）的难度级别
        :param even_link_speed_strength: 链路速度
        :param base_data_transmission_speed: 传输速度
        :param even_computation_power: 节点的计算能力
        :param is_malicious: 表示该节点是否为恶意节点
        :param noise_variance: 噪声方差
        :param check_signature: 用于验证交易或模型更新的数字签名是否有效
        :param not_resync_chain: 该节点是否允许链的重新同步
        :param malicious_updates_discount: 恶意更新的折扣率，对恶意节点的更新进行折扣处理
        :param knock_out_rounds: 联邦学习中被淘汰的轮次
        :param lazy_local_enterprise_knock_out_rounds: 节点懒惰被淘汰的轮次
        """
        self.idx = idx
        # deep learning variables
        self.train_ds = assigned_train_ds
        self.test_dl = assigned_test_dl
        self.local_batch_size = local_batch_size
        self.loss_func = loss_func
        self.network_stability = network_stability
        self.net = copy.deepcopy(net)  # 深度拷贝，对其任何修改不会影响到原始对象
        if opti == "SGD":
            # 使用随机梯度下降优化器
            self.opti = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        else:
            # 使用Adam优化器
            self.opti = optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.9, 0.9))
        self.dev = dev  # 设备 比如 cpu 或者 cuda:0
        # in real system, new data can come in, so train_dl should get reassigned before training when that happens
        self.train_dl = DataLoader(self.train_ds, batch_size=self.local_batch_size, shuffle=True)
        self.local_train_parameters = None  # 存储节点在本地训练过程中得到的模型参数
        self.initial_net_parameters = None  # 模型的初始参数
        self.global_parameters = None  # 全局模型参数
        # FedAnil+: Consortium_blockchain variables
        self.role = None  # 区块链网络中的角色，节点开始工作前，没有具体角色
        self.pow_difficulty = pow_difficulty  # 难度设置
        if even_link_speed_strength:
            self.link_speed = base_data_transmission_speed
        else:
            self.link_speed = random.random() * base_data_transmission_speed
        self.enterprises_dict = None  # 网络中节点的信息
        self.aio = False  # 是否使用异步I/O操作，False为不使用异步操作
        ''' simulating hardware equipment strength, such as good processors and RAM capacity. Following recorded times will be shrunk by this value of times
        # for local_enterprises, its update time
        # for miners, its PoW time
        # for validators, its validation time
        # might be able to simulate molopoly on computation power when there are block size limit, as faster enterprises' transactions will be accepted and verified first
        '''
        if even_computation_power:
            # 所有节点的计算能力是相同的，固定为1
            self.computation_power = 1
        else:
            # 每个节点随机分配一个从0到4之间的计算能力
            self.computation_power = random.randint(0, 4)
        # 存储与该节点直接相连的其他节点
        self.peer_list = set()
        # used in cross_verification and in the PoS
        # 该节点是否在线
        self.online = True
        # 跟踪该节点获得的奖励
        self.rewards = 0
        # FedAnil+: Consortium Blockchain
        self.consortium_blockchain = Consortium_Blockchain()
        # init key pair 生成密钥对
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()
        # black_list stores enterprise index rather than the object 黑名单与淘汰机制
        self.black_list = set()
        self.knock_out_rounds = knock_out_rounds
        self.lazy_local_enterprise_knock_out_rounds = lazy_local_enterprise_knock_out_rounds
        '''企业节点的更新状态'''
        self.local_enterprise_accuracy_accross_records = {}  # 存储本地企业节点在各个回合中的准确性记录
        self.has_added_block = False
        self.the_added_block = None
        self.is_malicious = is_malicious
        # if self.is_malicious:
        #	print(f"Malicious Node created {self.idx}")
        '''噪声方差与签名验证'''
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        '''链同步与恶意折扣'''
        self.not_resync_chain = not_resync_chain
        self.malicious_updates_discount = malicious_updates_discount
        '''记录本地企业和验证节点的行为'''
        # used to identify slow or lazy local_enterprises
        self.active_local_enterprise_record_by_round = {}  # 每个回合中活跃的本地企业节点
        self.untrustworthy_local_enterprises_record_by_comm_round = {}  # 每个回合中不可信赖的本地企业节点
        self.untrustworthy_validators_record_by_comm_round = {}  # 每个回合不可信赖的验证节点
        # for picking PoS legitimate blockd;bs
        # self.stake_tracker = {} # used some tricks in main.py for ease of programming
        # used to determine the slowest enterprise round end time to compare PoW with PoS round end time. If simulate under computation_power = 0, this may end up equaling infinity
        self.round_end_time = 0  # 回合结束时间，比较工作量证明POW和权益证明POS
        '''自适应霍夫曼编码和数据压缩'''
        self.adaptive_huffman_object = None
        self.AE = None
        self.coded_data_after_ac = None
        ''' For local_enterprises 本地企业的交易 '''
        self.local_updates_rewards_per_transaction = 0  # 本地企业每笔交易获得的奖励
        self.received_block_from_miner = None  # 存储从矿工节点接受到的区块
        self.accuracy_this_round = float('-inf')
        '''验证节点和矿工的关联'''
        self.local_enterprise_associated_validator = None
        self.local_enterprise_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        # FedAnil+: Training Models via Models Random Selection by Each Enterprise.
        # 模型类型的随机选择
        self.model_type = random.sample(m_type, random.randint(1, 3))
        ''' For validators '''
        self.validator_associated_local_enterprise_set = set()
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.validator_associated_miner = None
        # when validator directly accepts local_enterprises' updates
        self.unordered_arrival_time_accepted_local_enterprise_transactions = {}
        self.validator_accepted_broadcasted_local_enterprise_transactions = None or []
        self.final_transactions_queue_to_validate = {}
        self.post_validation_transactions_queue = None or []
        self.validator_threshold = validator_threshold
        self.validator_local_accuracy = None
        ''' For miners '''
        self.miner_associated_local_enterprise_set = set()
        self.miner_associated_validator_set = set()
        # dict cannot be added to set()
        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        # when miner directly accepts validators' updates
        self.unordered_arrival_time_accepted_validator_transactions = {}
        self.miner_accepted_broadcasted_validator_transactions = None or []
        self.final_candidate_transactions_queue_to_mine = {}
        self.block_generation_time_point = None
        self.unordered_propagated_block_processing_queue = {}  # pure simulation queue and does not exist in real distributed system
        ''' For malicious node '''
        self.variance_of_noises = None or []
        self.size_of_encoded_data = 0

    ''' Common Methods '''

    ''' setters '''

    def set_enterprises_dict_and_aio(self, enterprises_dict, aio):
        """
        传递一个企业相关信息的字典，异步I/O对象
        """
        self.enterprises_dict = enterprises_dict
        self.aio = aio

    def generate_rsa_key(self):
        """
        生成RSA密钥对
        存储模数，私钥，公钥
        """
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e

    def init_global_parameters(self):
        """
        初始化全局模型，两者在刚开始时是相同的
        state_dict() 获取神经网络模型的当前参数
        """
        self.initial_net_parameters = self.net.state_dict()
        self.global_parameters = self.net.state_dict()

    def return_global_parametesrs(self):
        """
        返回全局模型参数
        """
        return self.global_parameters

    def assign_role(self):
        """
        随机为对象分配一个角色
        生成一个介于0-2的随机整数 0，1，2
        模拟区块链中不同参与者
        """
        # equal probability
        role_choice = random.randint(0, 2)
        if role_choice == 0:
            self.role = "local_enterprise"
        elif role_choice == 1:
            self.role = "miner"
        else:
            self.role = "validator"

    # used for hard_assign
    def assign_miner_role(self):
        """
        区块链中指定角色为矿工
        """
        self.role = "miner"

    def assign_local_enterprise_role(self):
        """
        区块链中指定角色为本地企业
        """
        self.role = "local_enterprise"

    def assign_validator_role(self):
        """
        区块链中指定角色为验证者
        """
        self.role = "validator"

    ''' getters '''

    def return_idx(self):
        """
        返回索引标识符
        """
        return self.idx

    def return_rsa_pub_key(self):
        """
        返回包含RSA公钥的字典，模数和公钥
        """
        return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_peers(self):
        """
        返回一个包含与当前节点直接相连的其他节点（同伴节点）的列表 peer_list
        """
        return self.peer_list

    def return_role(self):
        """
        返回节点的角色 role，例如 "miner"、"validator" 或 "local_enterprise"
        """
        return self.role

    def is_online(self):
        """
        返回一个布尔值 online，表示节点是否在线
        """
        return self.online

    def return_is_malicious(self):
        """
        返回一个布尔值 is_malicious，表示节点是否为恶意节点
        """
        return self.is_malicious

    def return_black_list(self):
        """
        返回黑名单 black_list，黑名单上可能包含不可信任的节点标识符
        """
        return self.black_list

    # FedAnil+: Consortium Blockchain
    def return_consortium_blockchain_object(self):
        """
        返回与节点相关联的区块链对象 consortium_blockchain
        """
        return self.consortium_blockchain

    def return_stake(self):
        """
        返回节点获得的奖励 rewards
        """
        return self.rewards

    def return_computation_power(self):
        """
        返回节点的计算能力
        """
        return self.computation_power

    def return_the_added_block(self):
        """
        返回最近添加到区块链中的区块
        """
        return self.the_added_block

    def return_round_end_time(self):
        """
        返回当前通信轮次的结束时间
        """
        return self.round_end_time

    ''' functions '''

    def sign_msg(self, msg):
        """
        通过对消息生成哈希值并使用私钥对其进行加密来创建数字签名
        """
        # 传入消息的哈希值
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        # hash的private_key次方模modulus
        signature = pow(hash, self.private_key, self.modulus)
        return signature

    def add_peers(self, new_peers):
        """
        添加单个或多个同伴节点到一个节点的同伴列表中
        peer_list 集合
        添加一个 add
        添加列表或其他 update
        """
        if isinstance(new_peers, Enterprise):
            # 检查new_peers是否是Enterprise的实例
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)

    def remove_peers(self, peers_to_remove):
        """
        从节点的同伴列表中移除单个或多个节点
        discard 删除一个
        difference_update 删除列表或其他
        """
        if isinstance(peers_to_remove, Enterprise):
            self.peer_list.discard(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)

    def return_model_type(self, index):
        """
        返回 cnn glove resnet 中的元素，选择模型名称
        """
        return m_type[0]

    def online_switcher(self):
        """
        模拟节点的随机上线和下线，来模拟现实世界中网络连接的不稳定性
        """
        old_status = self.online
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            '''节点上线'''
            self.online = True
            # if back online, update peer and resync chain
            if old_status == False:
                print(f"{self.idx} goes back online.")
                # update peer list 更新同伴节点列表
                self.update_peer_list()
                # resync chain 重新同步区块链
                if self.pow_resync_chain():
                    # 重新同步成功，更新模型
                    self.update_model_after_chain_resync()
        else:
            '''节点下线'''
            self.online = False
            print(f"{self.idx} goes offline.")
        return self.online

    def update_peer_list(self):
        """
        更新同伴列表来维护节点的网络连接，确保节点能够与其他在线且非恶意的节点进行通信
        """
        print(f"\n{self.idx} - {self.role} is updating peer list...")
        old_peer_list = copy.copy(self.peer_list)
        online_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)
        # for online peers, suck in their peer list
        for online_peer in online_peers:
            self.add_peers(online_peer.return_peers())
        # remove itself from the peer_list if there is 从同伴列表中移除当前节点自身，避免自我引用
        self.remove_peers(self)
        # remove malicious peers
        removed_peers = []
        potential_malicious_peer_set = set()
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                potential_malicious_peer_set.add(peer)
        self.remove_peers(potential_malicious_peer_set)
        removed_peers.extend(potential_malicious_peer_set)
        # print updated peer result
        if old_peer_list == self.peer_list:
            print("Peer list NOT changed.")
        else:
            print("Peer list has been changed.")
            added_peers = self.peer_list.difference(old_peer_list)
            if potential_malicious_peer_set:
                print("These malicious peers are removed")
                for peer in removed_peers:
                    print(f"e_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            if added_peers:
                print("These peers are added")
                for peer in added_peers:
                    print(f"e_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print("Final peer list:")
            for peer in self.peer_list:
                print(f"e_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
            print()
        # WILL ALWAYS RETURN TRUE AS OFFLINE PEERS WON'T BE REMOVED ANY MORE, UNLESS ALL PEERS ARE Malicious Enterprises...but then it should not register with any other peer. Original purpose - if peer_list ends up empty, randomly register with another enterprise
        return False if not self.peer_list else True

    def check_pow_proof(self, block_to_check):
        """
        验证一个区块的工作量证明是否有效，确保区块的创建者已经完成了足够的计算工作
        返回值为布尔值 工作量证明有效 返回True 否则 返回 False
        """
        # remove its block hash(compute_hash() by default) to verify pow_proof as block hash was set after pow
        pow_proof = block_to_check.return_pow_proof()
        # print("pow_proof", pow_proof)
        # print("compute_hash", block_to_check.compute_hash())
        return pow_proof.startswith('0' * self.pow_difficulty) and pow_proof == block_to_check.compute_hash()

    def check_chain_validity(self, chain_to_check):
        """
        验证区块链的完整性和有效性
        检查链中的每个区块是否都有有效的工作量证明，并且每个区块都正确地链接到前一个区块
        """
        chain_len = chain_to_check.return_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            chain_to_check = chain_to_check.return_chain_structure()
            for block in chain_to_check[1:]:
                if self.check_pow_proof(block) and block.return_previous_block_hash() == chain_to_check[
                    chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True):
                    pass
                else:
                    return False
        return True

    def accumulate_chain_stake(self, chain_to_accumulate):
        """
        计算区块链中所有区块的累积权益
        """
        accumulated_stake = 0
        chain_to_accumulate = chain_to_accumulate.return_chain_structure()
        for block in chain_to_accumulate:
            accumulated_stake += self.enterprises_dict[block.return_mined_by()].return_stake()
        return accumulated_stake

    def resync_chain(self, mining_consensus):
        """
        在区块链网络中重新同步节点的区块链状态
        pow 工作量证明机制 通过算力来决定记账权 安全 去中心化 但是不高效，资源浪费多
        pos 权益证明机制 通过币龄来决定记账权 资源浪费少 但是容易中心化
        pow和pos都属于共识机制
        """
        if self.not_resync_chain:
            return  # temporary workaround to save GPU memory
        if mining_consensus == 'PoW':
            # 工作量证明 proof of work
            self.pow_resync_chain()
        else:
            # 权益证明 proof of stake
            self.pos_resync_chain()

    def pos_resync_chain(self):
        """
        采用POS权益证明共识机制的区块链网络中，用于重新同步节点的区块链状态
        找到一个累积权益最高的区块链，并将其设置为节点的本地区块链状态
        """
        print(
            f"{self.role} {self.idx} is looking for a chain with the highest accumulated miner's stake in the network...")
        highest_stake_chain = None  # 存储找到累积权益最高的区块链对象
        updated_from_peer = None  # 存储更新来源的节点索引
        # FedAnil+: Consortium Blockchain
        curr_chain_stake = self.accumulate_chain_stake(self.return_consortium_blockchain_object())  # 当前节点的区块链的累积权益
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_consortium_blockchain_object()  # 获取同伴节点的区块链对象
                peer_chain_stake = self.accumulate_chain_stake(peer_chain)  # 计算同伴节点区块链的累积权益
                if peer_chain_stake > curr_chain_stake:
                    if self.check_chain_validity(peer_chain):
                        print(
                            f"A chain from {peer.return_idx()} with total stake {peer_chain_stake} has been found (> currently compared chain stake {curr_chain_stake}) and verified.")
                        # Higher stake valid chain found!
                        curr_chain_stake = peer_chain_stake  # 更新最高的累积权益值
                        highest_stake_chain = peer_chain  # 更新找到的累积权益最高的区块链对象
                        updated_from_peer = peer.return_idx()  # 更新来源的节点索引
                    else:
                        # 区块链对象不有效，打印日志
                        print(
                            f"A chain from {peer.return_idx()} with higher stake has been found BUT NOT verified. Skipped this chain for syncing.")
        if highest_stake_chain:
            # compare chain difference
            highest_stake_chain_structure = highest_stake_chain.return_chain_structure()
            # need more efficient mechanism which is to reverse updates by # of blocks
            self.return_consortium_blockchain_object().replace_chain(
                highest_stake_chain_structure)  # 将当前节点的区块链替换为找到的累积权益最高的区块链
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")  # 区块链已从某个同伴节点重新同步。
            # return block_iter
            return True
        print("Chain not resynced.")
        return False

    def pow_resync_chain(self):
        """
        采用POW工作量证明共识机制的区块链网络中，重新同步节点的区块链状态
        找到一个长度最长的区块链，并将其设置为节点的本地区块链状态
        在PoW中，最长的链被认为是最安全的链
        """
        print(f"{self.role} {self.idx} is looking for a longer chain in the network...")
        longest_chain = None  # 存储找到的长度最长的区块链对象
        updated_from_peer = None  # 存储更新来源的节点索引
        curr_chain_len = self.return_consortium_blockchain_object().return_chain_length()  # 当前区块链的长度
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_consortium_blockchain_object()  # 获取同伴节点的区块链对象
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer_chain):
                        print(
                            f"A longer chain from {peer.return_idx()} with chain length {peer_chain.return_chain_length()} has been found (> currently compared chain length {curr_chain_len}) and verified.")
                        # Longer valid chain found!
                        curr_chain_len = peer_chain.return_chain_length()  # 更新当前最长的区块链长度
                        longest_chain = peer_chain  # 更新找到的长度最长的区块链对象
                        updated_from_peer = peer.return_idx()  # 更新来源的节点索引
                    else:
                        # 区块链对象不有效，打印日志
                        print(
                            f"A longer chain from {peer.return_idx()} has been found BUT NOT verified. Skipped this chain for syncing.")
        if longest_chain:
            # compare chain difference
            longest_chain_structure = longest_chain.return_chain_structure()
            # need more efficient mechanism which is to reverse updates by # of blocks
            self.return_consortium_blockchain_object().replace_chain(longest_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            # return block_iter
            return True
        print("Chain not resynced.")
        return False

    def load_data_by_index(self, index):
        """
        生成一个具有明显聚类结构的合成数据集。这些数据集可以用来测试聚类算法的性能
        生成样本点以及对应的真实标签
        """
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(
            n_samples=300, centers=centers, cluster_std=0.5, random_state=0
        )
        return X, labels_true

    def receive_rewards(self, rewards):
        """
        更新类实例中存储的奖励总额
        """
        self.rewards += rewards

    def verify_miner_transaction_by_signature(self, transaction_to_verify, miner_enterprise_idx):
        """
        验证矿工交易的有效性
        通过检查交易的数字签名来确保交易是由授权的矿工发起的
        """
        if miner_enterprise_idx in self.black_list:  # 判断矿工节点的索引是否在黑名单
            print(f"{miner_enterprise_idx} is in miner's blacklist. Transaction won't get verified.")
            return False
        if self.check_signature:
            transaction_before_signed = copy.deepcopy(transaction_to_verify)
            del transaction_before_signed["miner_signature"]
            # 从交易对象中获取矿工的 RSA 公钥的模数和公钥指数
            modulus = transaction_to_verify['miner_rsa_pub_key']["modulus"]
            pub_key = transaction_to_verify['miner_rsa_pub_key']["pub_key"]
            # 获取交易的数字签名
            signature = transaction_to_verify["miner_signature"]
            # verify 计算交易数据的 SHA-256 哈希值，并将其转换为整数
            hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                                  byteorder='big')
            # 使用矿工的公钥验证数字签名
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash == hashFromSignature:
                print(f"A transaction recorded by miner {miner_enterprise_idx} in the block is verified!")
                return True
            else:
                print(f"Signature invalid. Transaction recorded by {miner_enterprise_idx} is NOT verified.")
                return False
        else:
            # 不需要检查前面
            print(f"A transaction recorded by miner {miner_enterprise_idx} in the block is verified!")
            return True

    def verify_block(self, block_to_verify, sending_miner):
        """
        验证区块链中一个区块的有效性
        检查区块是否由授权的矿工创建，并且区块的工作量证明（Proof of Work, PoW）是否有效
        如果启用了签名验证，它还会检查区块的数字签名是否有效
        """
        if not self.online_switcher():
            # 节点离线
            print(f"{self.idx} goes offline when verifying a block")
            return False, False
        verification_time = time.time()
        mined_by = block_to_verify.return_mined_by()
        if sending_miner in self.black_list:
            # 检查发送区块的矿工是否在黑名单中
            print(
                f"The miner propagating/sending this block {sending_miner} is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        if mined_by in self.black_list:
            # 检查挖掘区块的矿工是否在黑名单中
            print(f"The miner {mined_by} mined this block is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        # check if the proof is valid(verify _block_hash). 验证区块的工作量证明是否有效
        if not self.check_pow_proof(block_to_verify):
            print(f"PoW proof of the block from miner {self.idx} is not verified.")
            return False, False
        # # check if miner's signature is valid 验证矿工的签名是否有效
        if self.check_signature:
            signature_dict = block_to_verify.return_miner_rsa_pub_key()  # 获取矿工的RSA公钥
            modulus = signature_dict["modulus"]
            pub_key = signature_dict["pub_key"]
            signature = block_to_verify.return_signature()
            # verify signature
            block_to_verify_before_sign = copy.deepcopy(block_to_verify)
            block_to_verify_before_sign.remove_signature_for_verification()
            hash = int.from_bytes(sha256(str(block_to_verify_before_sign.__dict__).encode('utf-8')).digest(),
                                  byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash != hashFromSignature:
                # 如果哈希值与从签名中恢复的哈希值不匹配
                print(
                    f"Signature of the block sent by miner {sending_miner} mined by miner {mined_by} is not verified by {self.role} {self.idx}.")
                return False, False
            # check previous hash based on own chain
            last_block = self.return_consortium_blockchain_object().return_last_block()
            if last_block is not None:
                # check if the previous_hash referred in the block and the hash of latest block in the chain match.
                last_block_hash = last_block.compute_hash(hash_entire_block=True)
                if block_to_verify.return_previous_block_hash() != last_block_hash:
                    print(
                        f"Block sent by miner {sending_miner} mined by miner {mined_by} has the previous hash recorded as {block_to_verify.return_previous_block_hash()}, but the last block's hash in chain is {last_block_hash}. This is possibly due to a forking event from last round. Block not verified and won't be added. Enterprise needs to resync chain next round.")
                    return False, False
        # All verifications done.
        print(f"Block accepted from miner {sending_miner} mined by {mined_by} has been verified by {self.idx}!")
        verification_time = (time.time() - verification_time) / self.computation_power
        return block_to_verify, verification_time

    # FedAnil+: get global model from blockchain
    def fetch_global_model(self, blockchain):
        """
        从一个区块链对象中获取全局模型参数
        最后一个区块代表区块链中最新的全局模型状态
        """
        try:
            last_block_in_blockchain = blockchain.return_last_block()
            transaction = last_block_in_blockchain["transaction"]
            self.local_model = transaction["gradients"]  # 包含了全局模型的梯度信息，当前节点的本地模型参数就更新为全局模型参数
        except:
            pass

    # FedAnil+: upload local model in compressed format
    def upload_local_model(self, compress_parameters):
        """
        将当前节点的本地模型参数（梯度信息）打包成一个新区块，并将其添加到联盟区块链的临时链中
        临时链会增加一个新的区块，这个区块包含最新的模型参数
        """
        new_transaction = {}
        new_transaction["gradients"] = compress_parameters
        new_block = Block(idx=-1, transactions=new_transaction, miner_rsa_pub_key=0)
        self.consortium_blockchain.new_local_block(new_block, self.coded_data_after_ac)

    # FedAnil+: fetch list ofcl all local models
    def fetch_local_models(self):
        """
        从区块链中检索与本地模型相关的数据
        """
        self.coded_data_after_ac = self.return_consortium_blockchain_object().return_last_cdata()
        return self.return_consortium_blockchain_object().return_local_chain()

    # FedAnil+: add consortium blockchain blockchain
    def add_block(self, block_to_add):
        """
        将一个创建好的区块添加到联盟区块链的正式链中
        区块包含交易信息，工作量证明等
        """
        self.return_consortium_blockchain_object().append_block(block_to_add)
        print(
            f"e_{self.idx.split('_')[-1]} - {self.role[0]} has appened a block to its chain. Chain length now - {self.return_consortium_blockchain_object().return_chain_length()}")
        # TODO delete has_added_block
        # self.has_added_block = True
        self.the_added_block = block_to_add
        return True

    # also accumulate rewards here
    def process_block(self, block_to_process, log_files_folder_path, conn, conn_cursor, when_resync=False):
        """
        处理一个传入的区块，并执行一系列操作，包括验证交易、识别恶意节点、收集奖励信息以及执行全局更新
        block_to_process：要处理的区块对象。
        log_files_folder_path：日志文件的存储路径。
        conn：数据库连接对象。
        conn_cursor：数据库游标对象，用于执行数据库操作。
        when_resync：一个布尔值，指示是否在重新同步区块链时处理区块
        """
        # collect usable updated params, malicious enterprises identification, get rewards and do local updates
        processing_time = time.time()
        if not self.online_switcher():
            print(
                # 可能需要设置一个标志代表某个块已经被处理
                f"{self.role} {self.idx} goes offline when processing the added block. Model not updated and rewards information not upgraded. Outdated information may be obtained by this node if it never resyncs to a different chain.")  # may need to set up a flag indicating if a block has been processed
        if block_to_process:
            mined_by = block_to_process.return_mined_by()
            if mined_by in self.black_list:
                # in this system black list is also consistent across enterprises as it is calculated based on the information on chain, but individual enterprise can decide its own validation/verification mechanisms and has its own
                print(
                    f"The added block is mined by miner {block_to_process.return_mined_by()}, which is in this enterprise's black list. Block will not be processed.")
            else:
                # process validator sig valid transactions
                # used to count positive and negative transactions local_enterprise by local_enterprise, select the transaction to do global update and identify potential malicious local_enterprise
                self_rewards_accumulator = 0  # 初始化一个累加器，用于累计节点在本轮中获得的奖励
                valid_transactions_records_by_local_enterprise = {}  # 存储每个本地企业的有效交易记录
                # 从区块中获取所有有效验证者签名的本地企业交易。
                valid_validator_sig_local_enterprise_transacitons_in_block = block_to_process.return_transactions()[
                    'valid_validator_sig_transacitons']
                comm_round = block_to_process.return_block_idx()  # 区块的通信轮次索引
                self.active_local_enterprise_record_by_round[comm_round] = set()  # 记录当前轮次中活跃的本地企业。
                # 遍历区块中所有有效验证者签名的本地企业交易。
                for valid_validator_sig_local_enterprise_transaciton in valid_validator_sig_local_enterprise_transacitons_in_block:
                    # verify miner's signature(miner does not get reward for receiving and aggregating)
                    # 验证矿工交易的签名是否有效
                    if self.verify_miner_transaction_by_signature(valid_validator_sig_local_enterprise_transaciton,
                                                                  mined_by):
                        local_enterprise_enterprise_idx = valid_validator_sig_local_enterprise_transaciton[
                            'local_enterprise_enterprise_idx']
                        self.active_local_enterprise_record_by_round[comm_round].add(local_enterprise_enterprise_idx)
                        # 如果本地企业索引不在有效交易记录中，则初始化其记录。
                        if not local_enterprise_enterprise_idx in valid_transactions_records_by_local_enterprise.keys():
                            valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx] = {}
                            valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                'positive_epochs'] = set()
                            valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                'negative_epochs'] = set()
                            valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                'all_valid_epochs'] = set()
                            valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                'finally_used_params'] = None
                        # epoch of this local_enterprise's local update
                        local_epoch_seq = valid_validator_sig_local_enterprise_transaciton[
                            'local_total_accumulated_epochs_this_round']
                        positive_direction_validators = valid_validator_sig_local_enterprise_transaciton[
                            'positive_direction_validators']
                        negative_direction_validators = valid_validator_sig_local_enterprise_transaciton[
                            'negative_direction_validators']
                        # all_direction_validators = valid_validator_sig_local_enterprise_transaciton['all_valid_epochs']
                        # FedAnil+: validation enterprise local update by all validators
                        # if len(positive_direction_validators) >= len(negative_direction_validators):
                        if len(negative_direction_validators) == 0:
                            # local_enterprise transaction can be used
                            valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                'positive_epochs'].add(local_epoch_seq)
                            valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                'all_valid_epochs'].add(local_epoch_seq)
                            # see if this is the latest epoch from this local_enterprise
                            if local_epoch_seq == max(
                                    valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                        'all_valid_epochs']):
                                valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                    'finally_used_params'] = valid_validator_sig_local_enterprise_transaciton[
                                    'local_updates_params']
                            # give rewards to this local_enterprise
                            if self.idx == local_enterprise_enterprise_idx:
                                self_rewards_accumulator += valid_validator_sig_local_enterprise_transaciton[
                                    'local_updates_rewards']
                        else:
                            if self.malicious_updates_discount:
                                # local_enterprise transaction voted negative and has to be applied for a discount
                                valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                    'negative_epochs'].add(local_epoch_seq)
                                valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                    'all_valid_epochs'].add(local_epoch_seq)
                                # see if this is the latest epoch from this local_enterprise
                                if local_epoch_seq == max(
                                        valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                            'all_valid_epochs']):
                                    # apply discount
                                    discounted_valid_validator_sig_local_enterprise_transaciton_local_updates_params = copy.deepcopy(
                                        valid_validator_sig_local_enterprise_transaciton['local_updates_params'])
                                    for var in discounted_valid_validator_sig_local_enterprise_transaciton_local_updates_params:
                                        discounted_valid_validator_sig_local_enterprise_transaciton_local_updates_params[
                                            var] *= self.malicious_updates_discount
                                    valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                        'finally_used_params'] = discounted_valid_validator_sig_local_enterprise_transaciton_local_updates_params
                                # local_enterprise receive discounted rewards for negative update
                                if self.idx == local_enterprise_enterprise_idx:
                                    self_rewards_accumulator += valid_validator_sig_local_enterprise_transaciton[
                                                                    'local_updates_rewards'] * self.malicious_updates_discount
                            else:
                                # discount specified as 0, local_enterprise transaction voted negative and cannot be used
                                valid_transactions_records_by_local_enterprise[local_enterprise_enterprise_idx][
                                    'negative_epochs'].add(local_epoch_seq)
                            # local_enterprise does not receive rewards for negative update
                        # give rewards to validators and the miner in this transaction
                        for validator_record in positive_direction_validators + negative_direction_validators:
                            if self.idx == validator_record['validator']:
                                self_rewards_accumulator += validator_record['validation_rewards']
                            if self.idx == validator_record['miner_enterprise_idx']:
                                self_rewards_accumulator += validator_record['miner_rewards_for_this_tx']
                    else:
                        print(
                            f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return

                # identify potentially malicious local_enterprise
                self.untrustworthy_local_enterprises_record_by_comm_round[comm_round] = set()
                for local_enterprise_idx, local_updates_direction_records in valid_transactions_records_by_local_enterprise.items():
                    if len(local_updates_direction_records['negative_epochs']) > len(
                            local_updates_direction_records['positive_epochs']):
                        self.untrustworthy_local_enterprises_record_by_comm_round[comm_round].add(local_enterprise_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_local_enterprises_record_by_comm_round.keys():
                                if local_enterprise_idx in self.untrustworthy_local_enterprises_record_by_comm_round[
                                    comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(local_enterprise_idx)
                            # is it right?
                            if when_resync:
                                msg_end = " when resyncing!\n"
                            else:
                                msg_end = "!\n"
                            if self.enterprises_dict[local_enterprise_idx].return_is_malicious():
                                msg = f"{self.idx} has successfully identified a malicious local_enterprise enterprise {local_enterprise_idx} in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/correctly_kicked_local_enterprises.txt",
                                          'a') as file:
                                    file.write(msg)
                                conn_cursor.execute(
                                    "INSERT INTO malicious_local_enterprises_log VALUES (?, ?, ?, ?, ?, ?)",
                                    (local_enterprise_idx, 1, self.idx, "", comm_round, when_resync))
                                conn.commit()
                            else:
                                msg = f"WARNING: {self.idx} has mistakenly regard {local_enterprise_idx} as a malicious local_enterprise enterprise in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/mistakenly_kicked_local_enterprises.txt",
                                          'a') as file:
                                    file.write(msg)
                                conn_cursor.execute(
                                    "INSERT INTO malicious_local_enterprises_log VALUES (?, ?, ?, ?, ?, ?)",
                                    (local_enterprise_idx, 0, "", self.idx, comm_round, when_resync))
                                conn.commit()
                            print(msg)

                        # cont = print("Press ENTER to continue")

                # identify potentially compromised validator
                self.untrustworthy_validators_record_by_comm_round[comm_round] = set()
                invalid_validator_sig_local_enterprise_transacitons_in_block = block_to_process.return_transactions()[
                    'invalid_validator_sig_transacitons']
                for invalid_validator_sig_local_enterprise_transaciton in invalid_validator_sig_local_enterprise_transacitons_in_block:
                    if self.verify_miner_transaction_by_signature(invalid_validator_sig_local_enterprise_transaciton,
                                                                  mined_by):
                        validator_enterprise_idx = invalid_validator_sig_local_enterprise_transaciton['validator']
                        self.untrustworthy_validators_record_by_comm_round[comm_round].add(validator_enterprise_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_validators_record_by_comm_round.keys():
                                if validator_enterprise_idx in self.untrustworthy_validators_record_by_comm_round[
                                    comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(validator_enterprise_idx)
                            print(
                                f"{validator_enterprise_idx} has been regarded as a compromised validator by {self.idx} in {comm_round}.")
                        # actually, we did not let validator do malicious thing if is_malicious=1 is set to this enterprise. In the submission of 2020/10, we only focus on catching malicious local_enterprise
                        # is it right?
                        # if when_resync:
                        #	 msg_end = " when resyncing!\n"
                        # else:
                        #	 msg_end = "!\n"
                        # if self.enterprises_dict[validator_enterprise_idx].return_is_malicious():
                        #	 msg = f"{self.idx} has successfully identified a compromised validator enterprise {validator_enterprise_idx} in comm_round {comm_round}{msg_end}"
                        #	 with open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'a') as file:
                        #		 file.write(msg)
                        # else:
                        #	 msg = f"WARNING: {self.idx} has mistakenly regard {validator_enterprise_idx} as a compromised validator enterprise in comm_round {comm_round}{msg_end}"
                        #	 with open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'a') as file:
                        #		 file.write(msg)
                        # print(msg)
                        # cont = print("Press ENTER to continue")
                    else:
                        print(
                            f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return
                    # give rewards to the miner in this transaction
                    if self.idx == invalid_validator_sig_local_enterprise_transaciton['miner_enterprise_idx']:
                        self_rewards_accumulator += invalid_validator_sig_local_enterprise_transaciton[
                            'miner_rewards_for_this_tx']
                # miner gets mining rewards
                if self.idx == mined_by:
                    self_rewards_accumulator += block_to_process.return_mining_rewards()
                # set received rewards this round based on info from this block
                self.receive_rewards(self_rewards_accumulator)
                print(
                    f"{self.role} {self.idx} has received total {self_rewards_accumulator} rewards for this comm round.")
                # collect usable local_enterprise updates and do global updates
                finally_used_local_params = []
                for local_enterprise_enterprise_idx, local_params_record in valid_transactions_records_by_local_enterprise.items():
                    if local_params_record['finally_used_params']:
                        # could be None
                        finally_used_local_params.append(
                            (local_enterprise_enterprise_idx, local_params_record['finally_used_params']))
                if self.online_switcher():
                    self.global_update(finally_used_local_params)
                else:
                    print(f"Unfortunately, {self.role} {self.idx} goes offline when it's doing global_updates.")
        processing_time = (time.time() - processing_time) / self.computation_power
        return processing_time

    def add_to_round_end_time(self, time_to_add):
        self.round_end_time += time_to_add

    def other_tasks_at_the_end_of_comm_round(self, this_comm_round, log_files_folder_path):
        self.kick_out_slow_or_lazy_local_enterprises(this_comm_round, log_files_folder_path)

    def kick_out_slow_or_lazy_local_enterprises(self, this_comm_round, log_files_folder_path):
        for enterprise in self.peer_list:
            if enterprise.return_role() == 'local_enterprise':
                if this_comm_round in self.active_local_enterprise_record_by_round.keys():
                    if not enterprise.return_idx() in self.active_local_enterprise_record_by_round[this_comm_round]:
                        not_active_accumulator = 1
                        # check if not active for the past (lazy_local_enterprise_knock_out_rounds - 1) rounds
                        for comm_round_to_check in range(
                                this_comm_round - self.lazy_local_enterprise_knock_out_rounds + 1, this_comm_round):
                            if comm_round_to_check in self.active_local_enterprise_record_by_round.keys():
                                if not enterprise.return_idx() in self.active_local_enterprise_record_by_round[
                                    comm_round_to_check]:
                                    not_active_accumulator += 1
                        if not_active_accumulator == self.lazy_local_enterprise_knock_out_rounds:
                            # kick out
                            self.black_list.add(enterprise.return_idx())
                            msg = f"local_enterprise {enterprise.return_idx()} has been regarded as a lazy local_enterprise by {self.idx} in comm_round {this_comm_round}.\n"
                            with open(f"{log_files_folder_path}/kicked_lazy_local_enterprises.txt", 'a') as file:
                                file.write(msg)
                else:
                    # this may happen when a enterprise is put into black list by every local_enterprise in a certain comm round
                    pass

    def update_model_after_chain_resync(self, log_files_folder_path, conn, conn_cursor):
        # reset global params to the initial weights of the net
        self.global_parameters = copy.deepcopy(self.initial_net_parameters)
        # in future version, develop efficient updating algorithm based on chain difference
        for block in self.return_consortium_blockchain_object().return_chain_structure():
            self.process_block(block, log_files_folder_path, conn, conn_cursor, when_resync=True)

    def return_pow_difficulty(self):
        return self.pow_difficulty

    def register_in_the_network(self, check_online=False):
        """
        模拟企业节点在网络中注册的过程
        """
        if self.aio:
            # 所有企业节点直接添加到当前企业节点的对等节点列表中，表示该企业节点与所有其他企业节点相连
            self.add_peers(set(self.enterprises_dict.values()))

        else:
            potential_registrars = set(self.enterprises_dict.values())
            # it cannot register with itself
            potential_registrars.discard(self)
            # pick a registrar
            registrar = random.sample(potential_registrars, 1)[0]
            if check_online:
                if not registrar.is_online():
                    online_registrars = set()
                    for registrar in potential_registrars:
                        if registrar.is_online():
                            online_registrars.add(registrar)
                    if not online_registrars:
                        return False
                    registrar = random.sample(online_registrars, 1)[0]
            # registrant add registrar to its peer list
            self.add_peers(registrar)
            # this enterprise sucks in registrar's peer list
            self.add_peers(registrar.return_peers())
            # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
            registrar.add_peers(self)
            return True

    ''' Local Enterprise '''

    def malicious_local_enterprise_add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                noise = self.noise_variance * torch.randn(m.weight.size())
                variance_of_noise = torch.var(noise)
                m.weight.add_(noise.to(self.dev))
                self.variance_of_noises.append(float(variance_of_noise))

    def malicious_local_enterprise_add_noise_to_datas(self, m):
        # done in DatasetLoad.py
        pass

    # TODO change to computation power
    # FedAnil+: Local Update
    def local_enterprise_local_update(self, rewards, log_files_folder_path_comm_round, comm_round, local_epochs=1):
        """
        模拟一个本地企业在联邦学习环境中进行本地模型训练(本地更新)的过程
        """
        # 打印本地企业正在执行本地更新的信息，包括它的索引、计算能力和链接速度
        print(
            f"Local Enterprise {self.idx} is doing local_update with computation power {self.computation_power} and link speed {round(self.link_speed, 3)} bytes/s")
        self.net.load_state_dict(self.global_parameters, strict=True)  # 加载全局模型参数
        # Total Computation Cost (Second)
        self.local_update_time = time.time()  # 记录本地更新开始的时间
        # local local_enterprise update by specified epochs
        # usually, if validator acception time is specified, local_epochs should be 1   如果指定了validator接受时间，local_epochs应该为1
        # logging maliciousness
        is_malicious_node = "M" if self.return_is_malicious() else "B"  # M表示恶意，B表示良性
        self.local_updates_rewards_per_transaction = 0  # 初始化本地更新的奖励变量
        # FedAnil+: Training the models that were selected. 本地模型训练,计算本地更新的奖励
        print(self.model_type)
        for mt in self.model_type:  # 模型类型随机选择 cnn，resnet，glove
            model_type_name = self.return_model_type(mt)  # 永远是cnn
            for epoch in range(local_epochs):
                for data, label in self.train_dl:
                    data, label = data.to(self.dev), label.to(self.dev)  # dev mps
                    preds = self.net(data, model_type_name)
                    loss = self.loss_func(preds, label)
                    loss.backward()
                    self.opti.step()
                    self.opti.zero_grad()
                    self.local_updates_rewards_per_transaction += rewards * (label.shape[0])

                # record accuracies to find good -vh  记录准确率，每个epoch结束后记录模型的准确率，并写入日志文件
                with open(
                        f"{log_files_folder_path_comm_round}/local_enterprise_{self.idx}_{is_malicious_node}_local_updating_accuracies_comm_{comm_round}.txt",
                        "a") as file:
                    file.write(
                        f"{self.return_idx()} epoch_{epoch + 1} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
                self.local_total_epoch += 1
        # FedAnil+: K-Medoids based Quantization K-Medoids量化
        self.net.kmedoids_update()
        # FedAnil+: Adaptive Huffman Encoding 自适应霍夫曼编码
        self.adaptive_huffman_encoding()
        # FedAnil+: Homomorphic Encryption 同态加密
        self.homomorphic_encryption()
        # local update done
        # Total Computation Cost (Second)
        try:
            self.local_update_time = (time.time() - self.local_update_time) / self.computation_power
        except:
            self.local_update_time = float('inf')
        # if self.is_malicious:
        # self.net.apply(self.malicious_local_enterprise_add_noise_to_weights)
        # print(f"malicious local_enterprise {self.idx} has added noise to its local updated weights before transmitting")
        # with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
        # file.write(f"{self.return_idx()} {self.return_role()} {is_malicious_node} noise variances: {self.variance_of_noises}\n")
        #

        # record accuracies to find good -vh
        with open(f"{log_files_folder_path_comm_round}/local_enterprise_final_local_accuracies_comm_{comm_round}.txt",
                  "a") as file:
            file.write(
                f"{self.return_idx()} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
        print(f"Done {local_epochs} epoch(s) and total {self.local_total_epoch} epochs")
        self.local_train_parameters = self.net.state_dict()
        self.upload_local_model(self.net.state_dict())
        return self.local_update_time

    # used to simulate time waste when local_enterprise goes offline during transmission to validator
    def waste_one_epoch_local_update_time(self, opti):
        if self.computation_power == 0:
            return float('inf'), None
        else:
            validation_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']
            # by default use SGD. Did not implement others
            if opti == 'SGD':
                validation_opti = optim.SGD(validation_net.parameters(), lr=currently_used_lr, momentum=0.9)
            else:
                validation_opti = optim.Adam(validation_net.parameters(), lr=currently_used_lr, betas=(0.9, 0.9))
            local_update_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = validation_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            return (time.time() - local_update_time) / self.computation_power, validation_net.state_dict()

    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy

    def return_accuracy_this_round(self):
        return self.accuracy_this_round

    def return_link_speed(self):
        return self.link_speed

    def return_local_updates_and_signature(self, comm_round):
        # local_total_accumulated_epochs_this_round also stands for the lastest_epoch_seq for this transaction(local params are calculated after this amount of local epochs in this round)
        # last_local_iteration(s)_spent_time may be recorded to determine calculating time? But what if nodes do not wish to disclose its computation power
        local_updates_dict = {'local_enterprise_enterprise_idx': self.idx, 'in_round_number': comm_round,
                              "local_updates_params": copy.deepcopy(self.local_train_parameters),
                              "local_updates_rewards": self.local_updates_rewards_per_transaction,
                              "local_iteration(s)_spent_time": self.local_update_time,
                              "local_total_accumulated_epochs_this_round": self.local_total_epoch,
                              "local_enterprise_rsa_pub_key": self.return_rsa_pub_key()}
        local_updates_dict["local_enterprise_signature"] = self.sign_msg(sorted(local_updates_dict.items()))
        return local_updates_dict

    def local_enterprise_reset_vars_for_new_round(self):
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.local_updates_rewards_per_transaction = 0
        self.has_added_block = False
        self.the_added_block = None
        self.local_enterprise_associated_validator = None
        self.local_enterprise_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        self.variance_of_noises.clear()
        self.round_end_time = 0

    def receive_block_from_miner(self, received_block, source_miner):
        if not (received_block.return_mined_by() in self.black_list or source_miner in self.black_list):
            self.received_block_from_miner = copy.deepcopy(received_block)
            print(
                f"{self.role} {self.idx} has received a new block from {source_miner} mined by {received_block.return_mined_by()}.")
        else:
            print(
                f"Either the block sending miner {source_miner} or the miner {received_block.return_mined_by()} mined this block is in local_enterprise {self.idx}'s black list. Block is not accepted.")

    def toss_received_block(self):
        self.received_block_from_miner = None

    def reset_last(self):
        global lastprc
        lastprc = 0

    def return_received_block_from_miner(self):
        return self.received_block_from_miner

    # FedAnil+: Total Accuracy (%)
    def validate_model_weights(self, weights_to_eval=None):
        """
        验证模型权重的有效性，在测试数据集上评估模型的准确率来实现
        """
        with torch.no_grad():  # 无需计算梯度，加载模型权重
            if weights_to_eval:
                self.net.load_state_dict(weights_to_eval, strict=True)
            else:
                self.net.load_state_dict(self.global_parameters, strict=True)
            # 初始化准确率统计变量
            sum_accu = 0
            num = 0
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)  # 预测概率最大的标签
                sum_accu += (preds == label).float().mean()  # 比较preds与labels的平均值得到准确率
                num += 1

            return sum_accu / num  # 平均准确率

    # FedAnil+: Global Update
    def global_update(self, local_update_params_potentially_to_be_used):  # global update
        # FedAnil+: get local updates from consortium blockchain
        self.get_local_params_by_local_enterprises = self.fetch_local_models()
        self.global_time = time.time()
        # FedAnil+: Adaptive Huffman Decoding
        self.adaptive_huffman_decoding()
        # FedAnil+: Calculating of Cosine Similarity: Object Handler
        cosine_similarity_operator = torch.nn.CosineSimilarity(dim=0)
        # filter local_params
        global lastprc, flcnt
        local_params_by_benign_local_enterprises = []
        for (local_enterprise_enterprise_idx, local_params) in local_update_params_potentially_to_be_used:
            if not local_enterprise_enterprise_idx in self.black_list:
                local_params_by_benign_local_enterprises.append(local_params)
            else:
                print(
                    f"global update skipped for a local_enterprise {local_enterprise_enterprise_idx} in {self.idx}'s black list")
        if local_params_by_benign_local_enterprises:
            nums_of_local_params = len(local_params_by_benign_local_enterprises)
            nums_of_local_param_len = len(local_params_by_benign_local_enterprises[0])
            similarity_matrix = np.zeros((nums_of_local_params, nums_of_local_param_len))
            i = 0
            sum_parameters = None
            for local_updates_params in local_params_by_benign_local_enterprises:
                j = 0
                for var in local_updates_params:
                    # FedAnil+: Calculating of Cosine Similarity: Distance of Local Models and Global Model in Prior Round
                    similarity = cosine_similarity_operator(local_params_by_benign_local_enterprises[i][var].view(-1),
                                                            self.global_parameters[var].view(-1))
                    if similarity > TRESHOLD1 and similarity < TRESHOLD2:
                        # sum_parameters[var] += local_updates_params[var]
                        similarity_matrix[i, j] = similarity
                    j += 1
                i += 1
            # FedAnil+: Affinity Propagation
            try:
                affinity_propagation = AffinityPropagation().fit(similarity_matrix)
                cluster_centers = affinity_propagation.cluster_centers_
                labels = affinity_propagation.labels_
                cluster_nums = np.zeros((len(cluster_centers), 1))
                sum_of_params_by_ap = np.zeros((len(cluster_centers), nums_of_local_param_len))
                for it in range(nums_of_local_params):
                    cluster_index = labels[it]
                    for var in local_params_by_benign_local_enterprises[it]:
                        sum_of_params_by_ap[cluster_index][var] += local_params_by_benign_local_enterprises[it][var]
                    cluster_nums[cluster_index] += 1
            except:
                pass
            # FedAnil+: FedAvg the gradients
            num_participants = len(local_params_by_benign_local_enterprises)
            sum_parameters = None
            for mt in m_type:
                for local_updates_params in local_params_by_benign_local_enterprises:
                    if sum_parameters is None:
                        sum_parameters = copy.deepcopy(local_updates_params)
                    else:
                        for var in sum_parameters:
                            if var.startswith(mt):
                                sum_parameters[var] += local_updates_params[var]
            for var in self.global_parameters:
                self.global_parameters[var] = (sum_parameters[var] / num_participants)
            print(f"global updates done by {self.idx}")
        else:
            print(f"There are no available local params for {self.idx} to perform global updates in this comm round.")

        self.global_time = time.time() - self.global_time

    ''' miner '''

    def request_to_download(self, block_to_download, requesting_time_point):
        print(
            f"miner {self.idx} is requesting its associated enterprises to download the block it just added to its chain")
        enterprises_in_association = self.miner_associated_validator_set.union(
            self.miner_associated_local_enterprise_set)
        for enterprise in enterprises_in_association:
            # theoratically, one enterprise is associated to a specific miner, so we don't have a miner_block_arrival_queue here
            if self.online_switcher() and enterprise.online_switcher():
                miner_link_speed = self.return_link_speed()
                enterprise_link_speed = enterprise.return_link_speed()
                lower_link_speed = enterprise_link_speed if enterprise_link_speed < miner_link_speed else miner_link_speed
                transmission_delay = getsizeof(str(block_to_download.__dict__)) / lower_link_speed
                verified_block, verification_time = enterprise.verify_block(block_to_download,
                                                                            block_to_download.return_mined_by())
                if verified_block:
                    # forgot to check for maliciousness of the block miner
                    enterprise.add_block(verified_block)
                enterprise.add_to_round_end_time(requesting_time_point + transmission_delay + verification_time)
            else:
                print(
                    f"Unfortunately, either miner {self.idx} or {enterprise.return_idx()} goes offline while processing this request-to-download block.")

    def propagated_the_block(self, propagating_time_point, block_to_propagate):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"{self.role} {self.idx} is propagating its mined block to {peer.return_role()} {peer.return_idx()}.")
                        if peer.online_switcher():
                            peer.accept_the_propagated_block(self, self.block_generation_time_point, block_to_propagate)
                    else:
                        print(
                            f"Destination miner {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Propagating skipped for this dest miner.")

    def accept_the_propagated_block(self, source_miner, source_miner_propagating_time_point, propagated_block):
        if not source_miner.return_idx() in self.black_list:
            source_miner_link_speed = source_miner.return_link_speed()
            this_miner_link_speed = self.link_speed
            lower_link_speed = this_miner_link_speed if this_miner_link_speed < source_miner_link_speed else source_miner_link_speed
            transmission_delay = getsizeof(str(propagated_block.__dict__)) / lower_link_speed
            self.unordered_propagated_block_processing_queue[
                source_miner_propagating_time_point + transmission_delay] = propagated_block
            print(
                f"{self.role} {self.idx} has accepted accepted a propagated block from miner {source_miner.return_idx()}")
        else:
            print(
                f"Source miner {source_miner.return_role()} {source_miner.return_idx()} is in {self.role} {self.idx}'s black list. Propagated block not accepted.")

    def add_propagated_block_to_processing_queue(self, arrival_time, propagated_block):
        self.unordered_propagated_block_processing_queue[arrival_time] = propagated_block

    def return_unordered_propagated_block_processing_queue(self):
        return self.unordered_propagated_block_processing_queue

    def return_associated_validators(self):
        return self.miner_associated_validator_set

    def return_miner_acception_wait_time(self):
        return self.miner_acception_wait_time

    def return_miner_accepted_transactions_size_limit(self):
        return self.miner_accepted_transactions_size_limit

    def return_miners_eligible_to_continue(self):
        miners_set = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                miners_set.add(peer)
        miners_set.add(self)
        return miners_set

    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions

    def verify_validator_transaction(self, transaction_to_verify):
        if self.computation_power == 0:
            print(f"miner {self.idx} has computation power 0 and will not be able to verify this transaction in time")
            return False, None
        else:
            transaction_validator_idx = transaction_to_verify['validation_done_by']
            if transaction_validator_idx in self.black_list:
                print(f"{transaction_validator_idx} is in miner's blacklist. Trasaction won't get verified.")
                return False, None
            verification_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_verify)
                del transaction_before_signed["validator_signature"]
                modulus = transaction_to_verify['validator_rsa_pub_key']["modulus"]
                pub_key = transaction_to_verify['validator_rsa_pub_key']["pub_key"]
                signature = transaction_to_verify["validator_signature"]
                # begin verification
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                                      byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(
                        f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                    verification_time = (time.time() - verification_time) / self.computation_power
                    return verification_time, True
                else:
                    print(f"Signature invalid. Transaction from validator {transaction_validator_idx} is NOT verified.")
                    return (time.time() - verification_time) / self.computation_power, False
            else:
                print(
                    f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                verification_time = (time.time() - verification_time) / self.computation_power
                return verification_time, True

    def sign_candidate_transaction(self, candidate_transaction):
        signing_time = time.time()
        candidate_transaction['miner_rsa_pub_key'] = self.return_rsa_pub_key()
        if 'miner_signature' in candidate_transaction.keys():
            del candidate_transaction['miner_signature']
        candidate_transaction["miner_signature"] = self.sign_msg(sorted(candidate_transaction.items()))
        signing_time = (time.time() - signing_time) / self.computation_power
        return signing_time

    def mine_block(self, candidate_block, rewards, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        pow_mined_block = self.proof_of_work(candidate_block)
        # pow_mined_block.set_mined_by(self.idx)
        pow_mined_block.set_mining_rewards(rewards)
        return pow_mined_block

    def proof_of_work(self, candidate_block, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        ''' Brute Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        # candidate_block.set_pow_difficulty(self.pow_difficulty)
        while not current_hash.startswith('0' * self.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other enterprises before adding the block
        # also set its hash as well. block_hash is the same as pow proof
        candidate_block.set_pow_proof(current_hash)
        return candidate_block

    def set_block_generation_time_point(self, block_generation_time_point):
        self.block_generation_time_point = block_generation_time_point

    def return_block_generation_time_point(self):
        return self.block_generation_time_point

    def receive_propagated_block(self, received_propagated_block):
        if not received_propagated_block.return_mined_by() in self.black_list:
            self.received_propagated_block = copy.deepcopy(received_propagated_block)
            print(
                f"Miner {self.idx} has received a propagated block from {received_propagated_block.return_mined_by()}.")
        else:
            print(
                f"Propagated block miner {received_propagated_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def receive_propagated_validator_block(self, received_propagated_validator_block):
        if not received_propagated_validator_block.return_mined_by() in self.black_list:
            self.received_propagated_validator_block = copy.deepcopy(received_propagated_validator_block)
            print(
                f"Miner {self.idx} has received a propagated validator block from {received_propagated_validator_block.return_mined_by()}.")
        else:
            print(
                f"Propagated validator block miner {received_propagated_validator_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def return_propagated_block(self):
        return self.received_propagated_block

    def return_propagated_validator_block(self):
        return self.received_propagated_validator_block

    def toss_propagated_block(self):
        self.received_propagated_block = None

    def toss_ropagated_validator_block(self):
        self.received_propagated_validator_block = None

    def miner_reset_vars_for_new_round(self):
        """
        重置矿工在新轮次开始时的状态变量
        """
        self.miner_associated_local_enterprise_set.clear()  # 清空与矿工关联的本地企业集合
        self.miner_associated_validator_set.clear()  # 清空与该矿工关联的验证者集合
        self.unconfirmmed_transactions.clear()  # 清空未确认的交易集合
        self.broadcasted_transactions.clear()  # 清空广播的交易集合
        # self.unconfirmmed_validator_transactions.clear()
        # self.validator_accepted_broadcasted_local_enterprise_transactions.clear()
        self.mined_block = None  # 重置矿工挖掘的区块为 None
        self.received_propagated_block = None  # 重置接收到的传播区块和验证者传播区块为 None
        self.received_propagated_validator_block = None
        self.has_added_block = False  # 标记矿工尚未添加区块
        self.the_added_block = None  # 重置添加的区块为 None
        self.unordered_arrival_time_accepted_validator_transactions.clear()  # 清空接收到的验证者交易的到达时间集合
        self.miner_accepted_broadcasted_validator_transactions.clear()  # 清空矿工接受的广播验证者交易集合
        self.block_generation_time_point = None  # 重置区块生成的时间点为 None
        #		self.block_to_add = None
        self.unordered_propagated_block_processing_queue.clear()  # 清空未排序的传播区块处理队列
        self.round_end_time = 0  # 将轮次结束时间重置为 0

    def set_unordered_arrival_time_accepted_validator_transactions(self,
                                                                   unordered_arrival_time_accepted_validator_transactions):
        self.unordered_arrival_time_accepted_validator_transactions = unordered_arrival_time_accepted_validator_transactions

    def return_unordered_arrival_time_accepted_validator_transactions(self):
        return self.unordered_arrival_time_accepted_validator_transactions

    def miner_broadcast_validator_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"miner {self.idx} is broadcasting received validator transactions to miner {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner = copy.copy(
                            self.unordered_arrival_time_accepted_validator_transactions)
                        # offline situation similar in validator_broadcast_local_enterprise_transactions()
                        for arrival_time, tx in self.unordered_arrival_time_accepted_validator_transactions.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del \
                                    final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner[
                                        arrival_time]
                        peer.accept_miner_broadcasted_validator_transactions(self,
                                                                             final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)
                        print(
                            f"miner {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)} validator transactions to miner {peer.return_idx()}.")
                    else:
                        print(
                            f"Destination miner {peer.return_idx()} is in miner {self.idx}'s black_list. broadcasting skipped for this dest miner.")

    def accept_miner_broadcasted_validator_transactions(self, source_enterprise,
                                                        unordered_transaction_arrival_queue_from_source_miner):
        # discard malicious node
        if not source_enterprise.return_idx() in self.black_list:
            self.miner_accepted_broadcasted_validator_transactions.append(
                {'source_enterprise_link_speed': source_enterprise.return_link_speed(),
                 'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_miner)})
            print(
                f"{self.role} {self.idx} has accepted validator transactions from {source_enterprise.return_role()} {source_enterprise.return_idx()}")
        else:
            print(
                f"Source miner {source_enterprise.return_role()} {source_enterprise.return_idx()} is in {self.role} {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_validator_transactions(self):
        return self.miner_accepted_broadcasted_validator_transactions

    def set_candidate_transactions_for_final_mining_queue(self, final_transactions_arrival_queue):
        self.final_candidate_transactions_queue_to_mine = final_transactions_arrival_queue

    def return_final_candidate_transactions_mining_queue(self):
        return self.final_candidate_transactions_queue_to_mine

    ''' validator '''

    def validator_reset_vars_for_new_round(self):
        self.validation_rewards_this_round = 0
        # self.accuracies_this_round = {}
        self.has_added_block = False
        self.the_added_block = None
        self.validator_associated_miner = None
        self.validator_local_accuracy = None
        self.validator_associated_local_enterprise_set.clear()
        # self.post_validation_transactions.clear()
        # self.broadcasted_post_validation_transactions.clear()
        self.unordered_arrival_time_accepted_local_enterprise_transactions.clear()
        self.final_transactions_queue_to_validate.clear()
        self.validator_accepted_broadcasted_local_enterprise_transactions.clear()
        self.post_validation_transactions_queue.clear()
        self.round_end_time = 0

    def add_post_validation_transaction_to_queue(self, transaction_to_add):
        self.post_validation_transactions_queue.append(transaction_to_add)

    def return_post_validation_transactions_queue(self):
        return self.post_validation_transactions_queue

    def return_online_local_enterprises(self):
        online_local_enterprises_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "local_enterprise":
                    online_local_enterprises_in_peer_list.add(peer)
        return online_local_enterprises_in_peer_list

    def return_validations_and_signature(self, comm_round):
        validation_transaction_dict = {'validator_enterprise_idx': self.idx, 'round_number': comm_round,
                                       'accuracies_this_round': copy.deepcopy(self.accuracies_this_round),
                                       'validation_effort_rewards': self.validation_rewards_this_round,
                                       "rsa_pub_key": self.return_rsa_pub_key()}
        validation_transaction_dict["signature"] = self.sign_msg(sorted(validation_transaction_dict.items()))
        return validation_transaction_dict

    def add_local_enterprise_to_association(self, local_enterprise_enterprise):
        if not local_enterprise_enterprise.return_idx() in self.black_list:
            self.associated_local_enterprise_set.add(local_enterprise_enterprise)
        else:
            print(
                f"WARNING: {local_enterprise_enterprise.return_idx()} in validator {self.idx}'s black list. Not added by the validator.")

    def associate_with_miner(self):
        miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == "miner":
                if not peer.return_idx() in self.black_list:
                    miners_in_peer_list.add(peer)
        if not miners_in_peer_list:
            return False
        self.validator_associated_miner = random.sample(miners_in_peer_list, 1)[0]
        return self.validator_associated_miner

    ''' miner and validator '''

    def add_enterprise_to_association(self, to_add_enterprise):
        """
        将一个企业添加到当前对象的关联集合中
        """
        # 若企业不在黑名单中,动态构建一个属性名，属性名为当前对象的角色
        if not to_add_enterprise.return_idx() in self.black_list:
            vars(self)[f'{self.role}_associated_{to_add_enterprise.return_role()}_set'].add(to_add_enterprise)
        else:
            print(
                f"WARNING: {to_add_enterprise.return_idx()} in {self.role} {self.idx}'s black list. Not added by the {self.role}.")

    def return_associated_local_enterprises(self):
        return vars(self)[f'{self.role}_associated_local_enterprise_set']

    def sign_block(self, block_to_sign):
        block_to_sign.set_signature(self.sign_msg(block_to_sign.__dict__))

    def add_unconfirmmed_transaction(self, unconfirmmed_transaction, souce_enterprise_idx):
        if not souce_enterprise_idx in self.black_list:
            self.unconfirmmed_transactions.append(copy.deepcopy(unconfirmmed_transaction))
            print(f"{souce_enterprise_idx}'s transaction has been recorded by {self.role} {self.idx}")
        else:
            print(
                f"Source enterprise {souce_enterprise_idx} is in the black list of {self.role} {self.idx}. Transaction has not been recorded.")

    def return_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    def broadcast_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == self.role:
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"{self.role} {self.idx} is broadcasting transactions to {peer.return_role()} {peer.return_idx()}.")
                        peer.accept_broadcasted_transactions(self, self.unconfirmmed_transactions)
                    else:
                        print(
                            f"Destination {peer.return_role()} {peer.return_idx()} is in {self.role} {self.idx}'s black_list. broadcasting skipped.")

    def accept_broadcasted_transactions(self, source_enterprise, broadcasted_transactions):
        # discard malicious node
        if not source_enterprise.return_idx() in self.black_list:
            self.broadcasted_transactions.append(copy.deepcopy(broadcasted_transactions))
            print(
                f"{self.role} {self.idx} has accepted transactions from {source_enterprise.return_role()} {source_enterprise.return_idx()}")
        else:
            print(
                f"Source {source_enterprise.return_role()} {source_enterprise.return_idx()} is in {self.role} {self.idx}'s black list. Transaction not accepted.")

    ''' local_enterprise and validator '''

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def return_mined_block(self):
        return self.mined_block

    def associate_with_enterprise(self, to_associate_enterprise_role):  # miner
        """
        为当前企业找到一个合适的对等企业，并建立关联，对等企业可能是miner或validator
        """
        # 尝试从当前企业的属性中获取一个已经关联的对等企业对象，使用vars(self)获取当前企业的所有属性
        to_associate_enterprise = vars(self)[f'{self.role}_associated_{to_associate_enterprise_role}']  # 第一次为None
        shuffled_peer_list = list(self.peer_list)
        random.shuffle(shuffled_peer_list)
        for peer in shuffled_peer_list:
            # select the first found eligible enterprise from a shuffled order 检查当前对等企业的角色是否与要关联的角色匹配，并且该对等企业是否在线
            if peer.return_role() == to_associate_enterprise_role and peer.is_online():
                if not peer.return_idx() in self.black_list:
                    to_associate_enterprise = peer
        if not to_associate_enterprise:
            # there is no enterprise matching the required associated role in this enterprise's peer list
            return False
        print(
            f"{self.role} {self.idx} associated with {to_associate_enterprise.return_role()} {to_associate_enterprise.return_idx()}")
        return to_associate_enterprise

    # FedAnil+: Homomorphic encryption
    def homomorphic_encryption(self):
        """
        对数据进行同态加密
        允许在加密数据上直接进行计算，不需要解密
        """
        # tenseal 创建同态加密上下文
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.generate_galois_keys()  # 生成生成伽罗瓦密钥
        context.global_scale = 2 ** 40  # 设置全局比例因子
        plain_data = self.coded_data_after_ac
        encrypted_data = {}
        try:
            for var in plain_data:
                encrypted_vars = {}
                labels_data = var["labels"]
                encrypted_vars["labels"] = ts.ckks_vector(context, labels_data)
                clusters_data = var["clusters"]
                encrypted_vars["clusters"] = ts.ckks_vector(context, clusters_data)
                encrypted_data[var] = encrypted_vars
        except:
            print("")
            print("")
            print(plain_data)
            print("")
        return encrypted_data

    # FedAnil+: Adaptive Huffman Encoding: EnCoding
    def adaptive_huffman_encoding(self):
        """
        对模型权重进行自适应霍夫曼编码，减少通信成本
        """
        # 获取K-Medoids聚类结果
        clusters_and_labels_from_kmedoids = self.net.kmedoids_update()
        coded_data = {}  # 存储编码后的数据
        byte_size = 0  # 累积编码数据的总字节大小
        # 遍历聚类结果
        for var in clusters_and_labels_from_kmedoids:
            data_to_encode = clusters_and_labels_from_kmedoids[var].labels_  # 提取变量的簇标签
            unique_labels_, counts_ = np.unique(data_to_encode, return_counts=True)  # 计算唯一标签和它们的计数
            count_of_any_elements = dict(zip(unique_labels_, counts_))  # 创建标签计数字典
            adaptive_huffman_dict = adaptive_huffman(count_of_any_elements)  # 构建自适应霍夫曼编码字典
            adaptive_huffman_object = bitarray()  # 存储位序列的数组
            adaptive_huffman_object.encode(adaptive_huffman_dict, data_to_encode)  # 使用霍夫曼编码字典编码簇标签
            one_var_dict = {}  # 存储编码后的标签数据
            one_var_dict["labels"] = adaptive_huffman_object  # 存储了与当前权重对应的编码后的霍夫曼位序列

            data_centers = clusters_and_labels_from_kmedoids[var].cluster_centers_  # 从K-Medoids聚类结果中提取权重的簇中心
            data_to_encode = [item for sublist in data_centers for item in sublist]  # 展平簇中心数据
            unique_labels_, counts_ = np.unique(data_to_encode,
                                                return_counts=True)  # 计算每个簇中心出现的次数，第一个为包含唯一簇中心的数组，第二个为每个簇中心的计数
            count_of_any_elements = dict(zip(unique_labels_, counts_))  # 将唯一的簇中心和计数合成一个字典，键为簇中心，值为对应的计数
            adaptive_huffman_dict = adaptive_huffman(count_of_any_elements)  # 构建哈夫曼编码字典
            adaptive_huffman_object = bitarray()  # 创建位数组对象
            adaptive_huffman_object.encode(adaptive_huffman_dict, data_to_encode)  # 使用霍夫曼编码字典编码簇中心
            one_var_dict["clusters"] = adaptive_huffman_object  # 存储了与当前权重对应的编码后的霍夫曼位序列
            coded_data[var] = one_var_dict  # var作为键，存储了标签和簇中心的编码的字典作为值
            # coded_data[var] = adaptive_huffman_object
            # Total Communication Cost (Byte): (initial gradients using the cluster head index + cluster head and their index)
            # 计算编码数据的字节大小,编码后的簇标签和簇中心的长度除以8(每个字节有8位)，加上簇中心的原始字节大小
            byte_size += math.ceil((len(one_var_dict["clusters"]) + len(one_var_dict["labels"])) / 8) + getsizeof(
                clusters_and_labels_from_kmedoids[var].cluster_centers_)
            # byte_size += math.ceil(len(adaptive_huffman_object) / 8)
            self.adaptive_huffman_object = adaptive_huffman_object  # 保存编码对象，按照这段代码来看，保存的是簇中心的编码结果
        self.size_of_encoded_data = byte_size  # 保存编码数据的总字节大小
        with open('./test/coed_data.txt', 'w') as f:
            for key, value in coded_data.items():
                f.write(f"{key}: {value}\n")
        return coded_data

    # FedAnil+: Adaptive Huffman Encoding: DeCoding
    def adaptive_huffman_decoding(self, huffman_coded_data=None):
        data_decoded_from_huffman = None
        huffman_coded_data = self.adaptive_huffman_object if not self.adaptive_huffman_object is None else bitarray()
        try:
            decoder_str = huffman_coded_data.decode()
            data_decoded_from_huffman = [int(decoder_str[i:i + 1]) for i in range(0, len(decoder_str))]
        except:
            pass
        return data_decoded_from_huffman

    ''' validator '''

    def set_unordered_arrival_time_accepted_local_enterprise_transactions(self, unordered_transaction_arrival_queue):
        self.unordered_arrival_time_accepted_local_enterprise_transactions = unordered_transaction_arrival_queue

    def return_unordered_arrival_time_accepted_local_enterprise_transactions(self):
        return self.unordered_arrival_time_accepted_local_enterprise_transactions

    def validator_broadcast_local_enterprise_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "validator":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"validator {self.idx} is broadcasting received validator transactions to validator {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_local_enterprise_transactions_for_dest_validator = copy.copy(
                            self.unordered_arrival_time_accepted_local_enterprise_transactions)
                        # if offline, it's like the broadcasted transaction was not received, so skip a transaction
                        for arrival_time, tx in self.unordered_arrival_time_accepted_local_enterprise_transactions.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del \
                                    final_broadcasting_unordered_arrival_time_accepted_local_enterprise_transactions_for_dest_validator[
                                        arrival_time]
                        # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received transactions(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the transactions itself received
                        peer.accept_validator_broadcasted_local_enterprise_transactions(self,
                                                                                        final_broadcasting_unordered_arrival_time_accepted_local_enterprise_transactions_for_dest_validator)
                        print(
                            f"validator {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_local_enterprise_transactions_for_dest_validator)} local_enterprise transactions to validator {peer.return_idx()}.")
                    else:
                        print(
                            f"Destination validator {peer.return_idx()} is in this validator {self.idx}'s black_list. broadcasting skipped for this dest validator.")

    def accept_validator_broadcasted_local_enterprise_transactions(self, source_validator,
                                                                   unordered_transaction_arrival_queue_from_source_validator):
        if not source_validator.return_idx() in self.black_list:
            self.validator_accepted_broadcasted_local_enterprise_transactions.append(
                {'source_validator_link_speed': source_validator.return_link_speed(),
                 'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_validator)})
            print(
                f"validator {self.idx} has accepted local_enterprise transactions from validator {source_validator.return_idx()}")
        else:
            print(
                f"Source validator {source_validator.return_idx()} is in validator {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_local_enterprise_transactions(self):
        return self.validator_accepted_broadcasted_local_enterprise_transactions

    def set_transaction_for_final_validating_queue(self, final_transactions_arrival_queue):
        self.final_transactions_queue_to_validate = final_transactions_arrival_queue

    def return_final_transactions_validating_queue(self):
        return self.final_transactions_queue_to_validate

    def validator_update_model_by_one_epoch_and_validate_local_accuracy(self, opti):
        # return time spent
        print(f"validator {self.idx} is performing one epoch of local update and validation")
        if self.computation_power == 0:
            print(f"validator {self.idx} has computation power 0 and will not be able to complete this validation")
            return float('inf')
        else:
            updated_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']
            # by default use SGD. Did not implement others
            if opti == 'SGD':
                validation_opti = optim.SGD(updated_net.parameters(), lr=currently_used_lr, momentum=0.9)
            else:
                validation_opti = optim.Adam(updated_net.parameters(), lr=currently_used_lr, betas=(0.9, 0.9))
            local_validation_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = updated_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            # validate by local test set
            with torch.no_grad():
                sum_accu = 0
                num = 0
                for data, label in self.test_dl:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = updated_net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
            self.validator_local_accuracy = sum_accu / num
            print(
                f"validator {self.idx} locally updated model has accuracy {self.validator_local_accuracy} on its local test set")
            return (time.time() - local_validation_time) / self.computation_power

    # TODO validator_threshold
    def validate_local_enterprise_transaction(self, transaction_to_validate, rewards, log_files_folder_path, comm_round,
                                              malicious_validator_on):
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if self.computation_power == 0:
            print(
                f"validator {self.idx} has computation power 0 and will not be able to validate this transaction in time")
            return False, False
        else:
            local_enterprise_transaction_enterprise_idx = transaction_to_validate['local_enterprise_enterprise_idx']
            if local_enterprise_transaction_enterprise_idx in self.black_list:
                print(
                    f"{local_enterprise_transaction_enterprise_idx} is in validator's blacklist. Trasaction won't get validated.")
                return False, False
            validation_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_validate)
                del transaction_before_signed["local_enterprise_signature"]
                modulus = transaction_to_validate['local_enterprise_rsa_pub_key']["modulus"]
                pub_key = transaction_to_validate['local_enterprise_rsa_pub_key']["pub_key"]
                signature = transaction_to_validate["local_enterprise_signature"]
                # begin validation
                # 1 - verify signature
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                                      byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(
                        f"Signature of transaction from local_enterprise {local_enterprise_transaction_enterprise_idx} is verified by validator {self.idx}!")
                    transaction_to_validate['local_enterprise_signature_valid'] = True
                else:
                    print(
                        f"Signature invalid. Transaction from local_enterprise {local_enterprise_transaction_enterprise_idx} does NOT pass verification.")
                    # will also add sig not verified transaction due to the validator's verification effort and its rewards needs to be recorded in the block
                    transaction_to_validate['local_enterprise_signature_valid'] = False
            else:
                print(
                    f"Signature of transaction from local_enterprise {local_enterprise_transaction_enterprise_idx} is verified by validator {self.idx}!")
                transaction_to_validate['local_enterprise_signature_valid'] = True
            # 2 - validate local_enterprise's local_updates_params if local_enterprise's signature is valid
            if transaction_to_validate['local_enterprise_signature_valid']:
                # accuracy validated by local_enterprise's update
                accuracy_by_local_enterprise_update_using_own_data = self.validate_model_weights(
                    transaction_to_validate["local_updates_params"])
                # if local_enterprise's accuracy larger, or lower but the difference falls within the validator threshold value, meaning local_enterprise's updated model favors validator's dataset, so their updates are in the same direction - True, otherwise False. We do not consider the accuracy gap so far, meaning if local_enterprise's update is way too good, it is still fine
                print(f'validator updated model accuracy - {self.validator_local_accuracy}')
                print(
                    f"After applying local_enterprise's update, model accuracy becomes - {accuracy_by_local_enterprise_update_using_own_data}")
                # record their accuracies and difference for choosing a good validator threshold
                is_malicious_validator = "M" if self.is_malicious else "B"
                with open(
                        f"{log_files_folder_path_comm_round}/validator_{self.idx}_{is_malicious_validator}_validation_records_comm_{comm_round}.txt",
                        "a") as file:
                    is_malicious_node = "M" if self.enterprises_dict[
                        local_enterprise_transaction_enterprise_idx].return_is_malicious() else "B"
                    file.write(
                        f"{accuracy_by_local_enterprise_update_using_own_data - self.validator_local_accuracy}: validator {self.return_idx()} {is_malicious_validator} in round {comm_round} evluating local_enterprise {local_enterprise_transaction_enterprise_idx}, diff = v_acc:{self.validator_local_accuracy} - w_acc:{accuracy_by_local_enterprise_update_using_own_data} {local_enterprise_transaction_enterprise_idx}_maliciousness: {is_malicious_node}\n")
                if accuracy_by_local_enterprise_update_using_own_data - self.validator_local_accuracy < self.validator_threshold * -1:
                    transaction_to_validate['update_direction'] = False
                    print(
                        f"NOTE: local_enterprise {local_enterprise_transaction_enterprise_idx}'s updates is deemed as suspiciously malicious by validator {self.idx}")
                    # is it right?
                    if not self.enterprises_dict[local_enterprise_transaction_enterprise_idx].return_is_malicious():
                        print(
                            f"Warning - {local_enterprise_transaction_enterprise_idx} is benign and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_local_enterprise_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_local_enterprise_update_using_own_data {accuracy_by_local_enterprise_update_using_own_data} , by validator {self.idx} on local_enterprise {local_enterprise_transaction_enterprise_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_negative_malicious_nodes_inside_caught.txt",
                                  'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_local_enterprise_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_local_enterprise_update_using_own_data {accuracy_by_local_enterprise_update_using_own_data} , by validator {self.idx} on local_enterprise {local_enterprise_transaction_enterprise_idx} in round {comm_round}\n")
                else:
                    transaction_to_validate['update_direction'] = True
                    print(
                        f"local_enterprise {local_enterprise_transaction_enterprise_idx}'s' updates is deemed as GOOD by validator {self.idx}")
                    # is it right?
                    if self.enterprises_dict[local_enterprise_transaction_enterprise_idx].return_is_malicious():
                        print(
                            f"Warning - {local_enterprise_transaction_enterprise_idx} is malicious and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt",
                                  'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_local_enterprise_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_local_enterprise_update_using_own_data {accuracy_by_local_enterprise_update_using_own_data} , by validator {self.idx} on local_enterprise {local_enterprise_transaction_enterprise_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_positive_good_nodes_inside_correct.txt", 'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_local_enterprise_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_local_enterprise_update_using_own_data {accuracy_by_local_enterprise_update_using_own_data} , by validator {self.idx} on local_enterprise {local_enterprise_transaction_enterprise_idx} in round {comm_round}\n")
                if self.is_malicious and malicious_validator_on:
                    old_voting = transaction_to_validate['update_direction']
                    transaction_to_validate['update_direction'] = not transaction_to_validate['update_direction']
                    with open(f"{log_files_folder_path_comm_round}/malicious_validator_log.txt", 'a') as file:
                        file.write(
                            f"malicious validator {self.idx} has flipped the voting of local_enterprise {local_enterprise_transaction_enterprise_idx} from {old_voting} to {transaction_to_validate['update_direction']} in round {comm_round}\n")
                transaction_to_validate['validation_rewards'] = rewards
            else:
                transaction_to_validate['update_direction'] = 'N/A'
                transaction_to_validate['validation_rewards'] = 0
            transaction_to_validate['validation_done_by'] = self.idx
            validation_time = (time.time() - validation_time) / self.computation_power
            transaction_to_validate['validation_time'] = validation_time
            transaction_to_validate['validator_rsa_pub_key'] = self.return_rsa_pub_key()
            # assume signing done in negligible time
            transaction_to_validate["validator_signature"] = self.sign_msg(sorted(transaction_to_validate.items()))
            return validation_time, transaction_to_validate


class EnterprisesInNetwork(object):
    """
    模拟和管理多个企业(客户端)，包含了企业的状态和行为，与区块链交互的逻辑
    """

    def __init__(self,
                 data_set_name,
                 is_iid,
                 batch_size,
                 learning_rate,
                 loss_func,
                 opti,
                 num_enterprises,
                 network_stability,
                 net,
                 dev,
                 knock_out_rounds,
                 lazy_local_enterprise_knock_out_rounds,
                 shard_test_data,
                 miner_acception_wait_time,
                 miner_accepted_transactions_size_limit,
                 validator_threshold,
                 pow_difficulty,
                 even_link_speed_strength,
                 base_data_transmission_speed,
                 even_computation_power,
                 malicious_updates_discount,
                 num_malicious,
                 noise_variance,
                 check_signature,
                 not_resync_chain):
        """
        :param data_set_name: 字符串，使用数据集名称 femnist
        :param is_iid: 布尔值，是否IID 0
        :param batch_size: 批量大小 64
        :param learning_rate: 学习率 0.01
        :param loss_func: 损失函数 F.cross_entropy
        :param opti: 优化器 SGD
        :param num_enterprises: 网络中企业数量 100
        :param network_stability: 网络稳定性 1.0
        :param net: 神经网络模型 CombineModel cnn
        :param dev: 设备 CPU或者GPU mps/cpu
        :param knock_out_rounds: 企业被踢出网络的轮次 5
        :param lazy_local_enterprise_knock_out_rounds: 懒惰的企业被踢出的轮次 10
        :param shard_test_data: 布尔值，是否分片测试数据 0
        :param miner_acception_wait_time: 矿工接受交易的等待时间 0.0
        :param miner_accepted_transactions_size_limit: 矿工接受交易大小限制 0.0
        :param validator_threshold: 验证器的阈值 0.08
        :param pow_difficulty: 工作量证明的难度 0
        :param even_link_speed_strength: 链接速度的均匀性 1
        :param base_data_transmission_speed: 基础数据传输速度 70000.0
        :param even_computation_power: 计算能力的均匀性 1
        :param malicious_updates_discount: 恶意更新的折扣 0.0
        :param num_malicious: 恶意企业的数量 3
        :param noise_variance: 注入的高斯噪声的方差 1
        :param check_signature: 是否检查签名 0
        :param not_resync_chain: 是否不重新同步链 1
        """
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.opti = opti
        self.num_enterprises = num_enterprises
        self.net = net
        self.dev = dev
        self.enterprises_set = {}  # 企业对象
        self.knock_out_rounds = knock_out_rounds
        self.lazy_local_enterprise_knock_out_rounds = lazy_local_enterprise_knock_out_rounds
        # self.test_data_loader = None
        self.default_network_stability = network_stability
        self.shard_test_data = shard_test_data
        self.even_link_speed_strength = even_link_speed_strength
        self.base_data_transmission_speed = base_data_transmission_speed
        self.even_computation_power = even_computation_power
        self.num_malicious = num_malicious
        self.malicious_updates_discount = malicious_updates_discount
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        # distribute dataset
        ''' validator '''
        self.validator_threshold = validator_threshold
        ''' miner '''
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        self.pow_difficulty = pow_difficulty
        ''' shard '''
        self.data_set_balanced_allocation()

    # distribute the dataset evenly to the enterprises
    def data_set_balanced_allocation(self):
        """
        将数据集均匀的分配给网络中的各个企业
        """
        # read dataset 读取数据集
        oarf_dataset = DatasetLoad(self.data_set_name, self.is_iid)  # femnist 0

        # perpare training data
        train_data = oarf_dataset.train_data  # (60000, 784)
        train_label = oarf_dataset.train_label  # (60000, 10)
        # shard dataset and distribute among enterprises 分割训练数据并分配给企业
        # shard train
        shard_size_train = oarf_dataset.train_data_size // self.num_enterprises // 2  # 60000 // 100 //2 =300
        shards_id_train = np.random.permutation(oarf_dataset.train_data_size // shard_size_train)  # 200 随机分配

        # perpare test data
        # 是否分片测试数据
        if not self.shard_test_data:
            # 不需要分片测试数据，直接将测试数据和标签转换为张量
            test_data = torch.tensor(oarf_dataset.test_data)  # (10000, 784)
            test_label = torch.argmax(torch.tensor(oarf_dataset.test_label), dim=1)  # (10000)
            test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=128, shuffle=False)
        else:
            test_data = oarf_dataset.test_data
            test_label = oarf_dataset.test_label
            # shard test
            shard_size_test = oarf_dataset.test_data_size // self.num_enterprises // 2  # 10000 // 100 //2 = 50
            shards_id_test = np.random.permutation(oarf_dataset.test_data_size // shard_size_test)  # 200

        # Collude Attack 处理恶意节点
        malicious_nodes_set = []
        if self.num_malicious:
            malicious_nodes_set = random.sample(range(self.num_enterprises), self.num_malicious)  # 在企业中随机选择3个企业当做恶意节点
        str_malicious_nodes = ""
        print('malicious_nodes_set:{}'.format(malicious_nodes_set))
        for mn in malicious_nodes_set:
            str_malicious_nodes += str(mn + 1) + ", "
        print(f"Malicious Enterprises: [{str_malicious_nodes}]\n")  # 恶意节点的索引
        for i in range(self.num_enterprises):
            is_malicious = False
            # make it more random by introducing two shards 获取当前企业要分配的两个数据片的ID
            shards_id_train1 = shards_id_train[i * 2]
            shards_id_train2 = shards_id_train[i * 2 + 1]
            # distribute training data 分配训练数据
            data_shards1 = train_data[
                           shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]  # （300，784）
            data_shards2 = train_data[
                           shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            label_shards1 = train_label[
                            shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]  # （300，10）
            label_shards2 = train_label[
                            shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            local_train_data, local_train_label = np.vstack((data_shards1, data_shards2)), np.vstack(
                (label_shards1, label_shards2))  # 在行上进行堆叠，e.g. (3,3)(3,3) (6,3)
            local_train_label = np.argmax(local_train_label, axis=1)
            if i in malicious_nodes_set:
                is_malicious = True
            # distribute test data 分配测试数据
            if self.shard_test_data:
                shards_id_test1 = shards_id_test[i * 2]
                shards_id_test2 = shards_id_test[i * 2 + 1]
                data_shards1 = test_data[
                               shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                data_shards2 = test_data[
                               shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                label_shards1 = test_label[
                                shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                label_shards2 = test_label[
                                shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                local_test_data, local_test_label = np.vstack((data_shards1, data_shards2)), np.vstack(
                    (label_shards1, label_shards2))
                local_test_label = torch.argmax(torch.tensor(local_test_label), dim=1)
                test_data_loader = DataLoader(
                    TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label)), batch_size=100,
                    shuffle=False)
                if is_malicious:
                    # add Gussian Noise 若企业是恶意的，为测试数据添加高斯噪声，模拟攻击
                    transform_method = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        AddGaussianNoise(mean=0, std=0.2 / len(malicious_nodes_set))
                    ])
                    test_data_loader = DataLoader(
                        TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label)), batch_size=128,
                        shuffle=False, transform=transform_method)

            # assign data to a enterprise and put in the enterprises set 创建企业对象并分配数据
            enterprise_idx = f'enterprise_{i + 1}'
            a_enterprise = Enterprise(idx=enterprise_idx,
                                      assigned_train_ds=TensorDataset(torch.tensor(local_train_data),
                                                                      torch.tensor(local_train_label)),
                                      assigned_test_dl=test_data_loader,
                                      local_batch_size=self.batch_size,
                                      learning_rate=self.learning_rate,
                                      loss_func=self.loss_func,
                                      opti=self.opti,
                                      network_stability=self.default_network_stability,
                                      net=self.net,
                                      dev=self.dev,
                                      miner_acception_wait_time=self.miner_acception_wait_time,
                                      miner_accepted_transactions_size_limit=self.miner_accepted_transactions_size_limit,
                                      validator_threshold=self.validator_threshold,
                                      pow_difficulty=self.pow_difficulty,
                                      even_link_speed_strength=self.even_link_speed_strength,
                                      base_data_transmission_speed=self.base_data_transmission_speed,
                                      even_computation_power=self.even_computation_power,
                                      is_malicious=is_malicious,
                                      noise_variance=self.noise_variance,
                                      check_signature=self.check_signature,
                                      not_resync_chain=self.not_resync_chain,
                                      malicious_updates_discount=self.malicious_updates_discount,
                                      knock_out_rounds=self.knock_out_rounds,
                                      lazy_local_enterprise_knock_out_rounds=self.lazy_local_enterprise_knock_out_rounds)
            # enterprise index starts from 1
            self.enterprises_set[enterprise_idx] = a_enterprise
            print(f"Sharding dataset to {enterprise_idx} done.")
        print(f"Sharding dataset done!")
