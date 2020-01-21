#!/usr/bin/python
#-*-coding:utf-8-*-
import json
from datetime import datetime
import os

## 根据身份证可以得到省份、具体到区县的地址信息、年龄、性别信息
## 需要输出对应信息，只需更改参数info为['province','address','sex','age']其中一个
#调用方法：from idcard import idcard_res, i = idcard_res(), i.find('612129198106120810')_
class idcard_res():
    def __init__(self):
        loc = os.path.dirname(__file__)
        data_loc = os.path.join(loc, 'id')
        with open(data_loc, 'r+') as f:
            da = f.read()
            self.idcard_addr = {d.split()[0]:d.split()[1] for d in da.splitlines()}

    def old_idcard_tran(self,idcard):
        if not isinstance(idcard,str):
            idcard = str(idcard)
        if idcard in ['nan','None','NAN','']:
            print('{} 身份证传入为空'.format(idcard))
            return None
        card_length = len(idcard)
        if card_length == 15:
            idcard = idcard[:6] + '19' + idcard[6:] + '0'
        elif card_length == 18:
            pass
        else:
            print('{} 身份证位数有误'.format(idcard))
            return None
        return idcard

    def get_sex(self, idcard):
        sex_code = int(idcard[-2])
        if sex_code % 2 == 1:
            return 1
        else:
            return 0

    def get_province(self, idcard):
        province_code = idcard[:2] + '0000'
        province = self.idcard_addr.get(province_code)
        return province

    def get_addr(self, idcard):
        addr_code = idcard[:6]
        addr_ls = self.idcard_addr.get(addr_code)
        if addr_ls:
            return addr_ls
        else:
            return None

    def get_age(self,idcard):
        birth_year = int(idcard[6:10])
        year_now = datetime.now().year
        age = year_now - birth_year
        return age

    def find(self,idcard,info = 'address'):
        '''
        :param info: 可选['province','address','sex','age']
        :return: 输出信息
        '''
        idcard = self.old_idcard_tran(idcard)
        if idcard:
            if info == 'address':
                addr = self.get_addr(idcard)
                if addr:
                    return addr
                else:
                    print('{} 获取不到地址信息'.format(idcard))
                    return None
            elif info == 'sex':
                sex = self.get_sex(idcard)
                return sex
            elif info == 'age':
                age = self.get_age(idcard)
                return age
            elif info == 'province':
                province = self.get_province(idcard)
                return province
            else:
                addr = self.get_addr(idcard)
                sex = self.get_sex(idcard)
                age = self.get_age(idcard)
                return addr,sex,age
        else:
            return None

if __name__ == '__main__':
    id_info = idcard_res()
    # 默认输出地址信息,可选输出省份、性比、年龄
    print(id_info.find('612129198106120810','address'))
    print(id_info.find('612129198106120810','province'))
    print(id_info.find('612129198106120810','sex'))
    print(id_info.find('612129198106120810','age'))