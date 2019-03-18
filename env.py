import numpy as np
import multi

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

mytable={0:'AA',1:'AB',2:'AC',3:'AD',4:'AE',5:'AF',6:'AG',7:'AH',8:'AI',9:'AJ',10:'AK',
          11:'AL',12:'AM',13:'AN',14:'AO',15:'AP',16:'AQ',17:'AR',18:'AS',19:'BA',20:'BB',
          21:'BC',22:'BD',23:'BE',24:'BF',25:'BG',26:'BH',27:'BI',28:'BJ',29:'BK',30:'BL',
          31:'BM',32:'BN',33:'BO',34:'BP',35:'BQ',36:'BR',37:'BS',38:'CA',39:'CB',40:'CC',
          41:'CD',42:'CE',43:'CF',44:'CG',45:'CH',46:'CI',47:'CJ',48:'CK',49:'CL',50:'CM',
          51:'CN',52:'CO',53:'CP',54:'CQ',55:'CR',56:'CS',57:'DA',58:'DB',59:'DC',60:'DD',
          61:'DE',62:'DF',63:'DG',64:'DH',65:'DI',66:'DJ',67:'DK',68:'DL',69:'DM',70:'DN',
          71:'DO',72:'DP',73:'DQ',74:'DR',75:'DS',76:'EA',77:'EB',78:'EC',79:'ED',80:'EE',
          81:'EF',82:'EG',83:'EH',84:'EI',85:'EJ',86:'EK',87:'EL',88:'EM',89:'EN',90:'EO',
          91:'EP',92:'EQ',93:'ER',94:'ES',95:'FA',96:'FB',97:'FC',98:'FD',99:'FE',100:'FF',
          101:'FG',102:'FH',103:'FI',104:'FJ',105:'FK',106:'FL',107:'FM',108:'FN',109:'FO',110:'FP',
          111:'FQ',112:'FR',113:'FS',114:'GA',115:'GB',116:'GC',117:'GD',118:'GE',119:'GF',120:'GG',
          121:'GH',122:'GI',123:'GJ',124:'GK',125:'GL',126:'GM',127:'GN',128:'GO',129:'GP',130:'GQ',
          131:'GR',132:'GS',133:'HA',134:'HB',135:'HC',136:'HD',137:'HE',138:'HF',139:'HG',140:'HH',
          141:'HI',142:'HJ',143:'HK',144:'HL',145:'HM',146:'HN',147:'HO',148:'HP',149:'HQ',150:'HR',
          151:'HS',152:'IA',153:'IB',154:'IC',155:'ID',156:'IE',157:'IF',158:'IG',159:'IH',160:'II',
          161:'IJ',162:'IK',163:'IL',164:'IM',165:'IN',166:'IO',167:'IP',168:'IQ',169:'IR',170:'IS',
          171:'JA',172:'JB',173:'JC',174:'JD',175:'JE',176:'JF',177:'JG',178:'JH',179:'JI',180:'JJ',
          181:'JK',182:'JL',183:'JM',184:'JN',185:'JO',186:'JP',187:'JQ',188:'JR',189:'JS',190:'KA',
          191:'KB',192:'KC',193:'KD',194:'KE',195:'KF',196:'KG',197:'KH',198:'KI',199:'KJ',200:'KK',
          201:'KL',202:'KM',203:'KN',204:'KO',205:'KP',206:'KQ',207:'KR',208:'KS',209:'LA',210:'LB',
          211:'LC',212:'LD',213:'LE',214:'LF',215:'LG',216:'LH',217:'LI',218:'LJ',219:'LK',220:'LL',
          221:'LM',222:'LN',223:'LO',224:'LP',225:'LQ',226:'LR',227:'LS',228:'MA',229:'MB',230:'MC',
          231:'MD',232:'ME',233:'MF',234:'MG',235:'MH',236:'MI',237:'MJ',238:'MK',239:'ML',240:'MM',
          241:'MN',242:'MO',243:'MP',244:'MQ',245:'MR',246:'MS',247:'NA',248:'NB',249:'NC',250:'ND',
          251:'NE',252:'NF',253:'NG',254:'NH',255:'NI',256:'NJ',257:'NK',258:'NL',259:'NM',260:'NN',
          261:'NO',262:'NP',263:'NQ',264:'NR',265:'NS',266:'OA',267:'OB',268:'OC',269:'OD',270:'OE',
          271:'OF',272:'OG',273:'OH',274:'OI',275:'OJ',276:'OK',277:'OL',278:'OM',279:'ON',280:'OO',
          281:'OP',282:'OQ',283:'OR',284:'OS',285:'PA',286:'PB',287:'PC',288:'PD',289:'PE',290:'PF',
          291:'PG',292:'PH',293:'PI',294:'PJ',295:'PK',296:'PL',297:'PM',298:'PN',299:'PO',300:'PP',
          301:'PQ',302:'PR',303:'PS',304:'QA',305:'QB',306:'QC',307:'QD',308:'QE',309:'QF',310:'QG',
          311:'QH',312:'QI',313:'QJ',314:'QK',315:'QL',316:'QM',317:'QN',318:'QO',319:'QP',320:'QQ',
          321:'QR',322:'QS',323:'RA',324:'RB',325:'RC',326:'RD',327:'RE',328:'RF',329:'RG',330:'RH',
          331:'RI',332:'RJ',333:'RK',334:'RL',335:'RM',336:'RN',337:'RO',338:'RP',339:'RQ',340:'RR',
          341:'RS',342:'SA',343:'SB',344:'SC',345:'SD',346:'SE',347:'SF',348:'SG',349:'SH',350:'SI',
          351:'SJ',352:'SK',353:'SL',354:'SM',355:'SN',356:'SO',357:'SP',358:'SQ',359:'SR',360:'SS',}

class cheese(object):
    def __init__(self):
        super(cheese, self).__init__()
        self.action_space = [i for i in range(361)]#补充
        self.n_actions = len(self.action_space)
        self.n_features =19*19*16
        self.observation=np.zeros(361) #第一种表示方式
        self.trans_observation=''  #表示当前的状态表示，第二种表示方式
        self.trans_2_observation='' #这里需要更改
        self.done=0
        self.reward=0


    #程序从头开始，清空棋盘，用于每一局从头开始，返回一个当前没有下棋的局面
    def reset(self):
        self.observation = np.zeros(361)
        self.trans_observation=''
        self.trans_2_observation=''#这里需要更改
        return self.trans_2_observation


    # #得到当前状态下，行使上面所选择得动作，返回下一个状态表达，回报，和是否结束
    # def step_s(self, action, pos):
    #     #用数组表示的
    #     if pos==1:#1代表的是进攻方   0代表防守方
    #         self.observation[action]=1
    #     else:
    #         self.observation[action]=-1
    #     #用字符床表示的，返回
    #     self.trans_observation+=mytable[action]
    #
    #
    #     return self.trans_observation
    #
    # def step_rd(self,action,pos):
    #     filename=str
    #
    # def trans_2(self,trans_observation,n):
    #     multi.interact(trans_observation,n)
    #     filename=str(n)+'.txt'
    #     #将发回的文件内容，写入到m_flush中
    #     m_flush=open(filename,'r').readlines()
    #     print(m_flush[0])
    #
    #     #下面是读取棋谱的操作


    def step(self,action,pos,n):

        ########### 状态更新阶段 #############
        # 表示1
        if pos == 1:  # 1代表的是进攻方   0代表防守方
            self.observation[action] = 1
        else:
            self.observation[action] = -1
        # 表示2
        self.trans_observation += mytable[action]
        # 表示3
        # 将2传给后得到
        multi.interact(self.trans_observation,n)
        filename=str(n)+'.txt'
        #将发回的文件内容，写入到m_flush中
        m_flush=open(filename,'r').readlines()
        self.trans_2_observation=m_flush
        #这里一会进行添加



        ############ 价值和是否完成 ###############
        self.reward=m_flush
        self.done=0
        #这里一会进行添加


# if __name__=='__main__':
#     a=cheese()
#     a.reset()
#     trans_observation=a.step_s(5,1)
#     trans_observation = a.step_s(7, 1)
#     #print(type(trans_observation))
#     a.trans_2(trans_observation,5)







