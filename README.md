# Introduction
Panzer, forked from [Trilinos](https://github.com/trilinos/Trilinos) package of Panzer, is a toolkit that provides a multiphysics assembly engine for solving large-scale systems of partial differential equations. 

This package of Panzer is developed for the purpose of supporting open source multiphysics FEM software [SiYuan](https://gitlab.com/hillyuan/siyuan). Those 
defectives of original panzer, which are to be used by SiYuan, such as
- Different types of element are not allowed
- Second order time derivatives are not considered

are to be developed/modified in this package.


# Usage

The current package of Panzer should be configured and compiled with Trilinos. You could download official Trilinos from https://github.com/trilinos/Trilinos and Panzer here and then modify line 134 of the file PackagesList.cmake of Trilinos from
```
Panzer                packages/panzer                   PT 
```
to
```
Panzer               <folder of current package>        PT 
```
The Trilinos could then be compiled with current package of Panzer. Pls check relevent introduction of Trilinos for details.

# License

BSD 3-Clause license. Pls see licensing declaration written in the top of each files for details.