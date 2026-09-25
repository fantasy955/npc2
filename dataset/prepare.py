# -*- coding: utf-8 -*-'
import os

txt = open('list/visda/train.txt', 'r')
txt_new = open('tmp.txt', 'w')

for line in txt.readlines():
    new_line = 'train/' + line
    txt_new.write(new_line)
txt_new.close()
txt.close()