import requests
from lxml import etree
import re
from fastbook import *
from fastai.vision.widgets import *


# 获取全部图片url
def parse_images_bing(o, i):
    url = 'https://cn.bing.com/images/async?q='+o+' bear&first=' + \
        str(i)+'&count=35&relp=35&scenario=ImageBasicHover&datsrc=N_I&layout=RowBased&mmasync=1'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding
    data = response.content.decode('utf-8', 'ignore')
    html = etree.HTML(data)
    conda_list = html.xpath('//a[@class="iusc"]/@m')
    all_url = []    # 保存全部图片的url
    for i in conda_list:
        img_url = re.search('"murl":"(.*?)"', i).group(1)
        all_url.append(img_url)
    return all_url


# 爬取图片函数
def crawler_images_bing(bear_types, path):
    if not path.exists():
        path.mkdir()
        for o in bear_types:
            dest = (path/o)
            dest.mkdir(exist_ok=True)
            for i in range(0, 180, 35):
                # 解析页面获取图片url
                img_data = parse_images_bing(o, i)
                # 下载图片
                download_images(dest, urls=img_data)
