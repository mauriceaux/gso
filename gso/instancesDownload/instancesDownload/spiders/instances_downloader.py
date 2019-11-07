#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:38:34 2019

@author: mauri
"""

import scrapy


class InstancesDownloader(scrapy.Spider):
    name = "downloader"

    def start_requests(self):
        urls = [
            'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        print(response.url)
#        exit()
        for a in response.xpath('//a[@href]/@href'):
            urlDownload = 'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/' + a.extract()
#            print(urlDownload)
            if urlDownload.endswith('.txt'):
                yield scrapy.Request(urlDownload, callback=self.save_txt)

    def save_txt(self, response):
        path = response.url.split('/')[-1]
        with open(path, 'wb') as f:
            f.write(response.body)