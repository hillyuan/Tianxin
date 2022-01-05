# Introduction
Tianxin, forked from [Trilinos](https://github.com/trilinos/Trilinos) package of Panzer, is a toolkit that provides a multiphysics assembly engine for solving large-scale systems of partial differential equations. 

Tianxin is developed for the purpose of supporting open source multiphysics FEM software [SiYuan](https://gitlab.com/hillyuan/siyuan). Those defectives of original software package Panzer, which are to be used by SiYuan, such as
- Different types of element are not allowed
- Second order time derivatives are not considered

are to be developed/modified in this package.


# Usage

Tianxin should be configured and compiled together with Trilinos. You should download official Trilinos from https://github.com/trilinos/Trilinos and Tianxin here and then modify line 134 of the file PackagesList.cmake of Trilinos from
```
Panzer                packages/panzer                   PT 
```
to
```
Tianxin               <folder of current package>        PT 
```
The Trilinos could then be compiled with current package of Tianxin. Pls check relevent instruction of Trilinos for details.

# License

BSD 3-Clause license. Pls see licensing declaration written in the top of each files for details.