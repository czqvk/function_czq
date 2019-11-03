# @Author  : Czq
# @Time    : 2019/10/14 16:45
# @File    : feature_tran.py

from datetime import datetime

#根据身份证衍生省份性别年龄特征
def iden_tran(iden_id = None,feat = 'p'):
    '''
    :param iden_id:身份证,只能以18位传入
    :param feat: ['p','s','a']
    :return:对应特征返回
    '''
    if len(iden_id) != 18:
        return None
    else:
        if feat == 'p':
            paper_pro = ['广东','广西','云南','福建','江西','湖南','贵州','四川','西藏','青海','新疆','宁夏','甘肃','山西','陕西','河南','湖北','河北',
                         '重庆', '安徽', '浙江','上海','江苏','山东','天津','北京','内蒙古','辽宁','吉林','黑龙江','海南','台湾','香港','澳门']
            id_pro = ['44','45','53','35','36','43','52','51','54','63','65','64','62','14','61','41','42','13','50',
                         '34', '33', '31','32','37','12','11','15','21','22','23','40','71','81','91']
            id_pro_dic = {x:y for x,y in zip(id_pro,paper_pro)}
            province = id_pro_dic.get(iden_id[:2])
            return province
        elif feat == 's':
            return 1 if int(iden_id[-2]) % 2 ==1 else 0
        elif feat == 'a':
            birth_year = int(iden_id[6:10])
            year_now = datetime.now().year
            age = year_now - birth_year
            return age
        else:
            print('check your feature_input,please input \'p\' or \'s\' or \'a\' ')
