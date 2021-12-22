#!/bin/bash
echo $'\n---PART 1---'
echo $'\nES - Part 1'
python evalResult.py inputData/ES/dev.out ES/dev.p1.out
echo $'\nRU - Part 1'
python evalResult.py inputData/RU/dev.out RU/dev.p1.out

echo $'\n---PART 2---'
echo $'\nES - Part 2'
python evalResult.py inputData/ES/dev.out ES/dev.p2.out
echo $'\nRU - Part 2'
python evalResult.py inputData/RU/dev.out RU/dev.p2.out

echo $'\n---PART 3---'
echo $'\nES - Part 3'
python evalResult.py inputData/ES/dev.out ES/dev.p3.out
echo $'\nRU - Part 3'
python evalResult.py inputData/RU/dev.out RU/dev.p3.out

echo $'\n---PART 4---'
echo $'\nES - Part 4'
python evalResult.py inputData/ES/dev.out ES/dev.p4.out
echo $'\nRU - Part 4'
python evalResult.py inputData/RU/dev.out RU/dev.p4.out
