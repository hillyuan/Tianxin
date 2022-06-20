// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) Xi Yuan
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ***********************************************************************
// @HEADER

#ifndef _TIANXIN_PARAMETER_HPP
#define _TIANXIN_PARAMETER_HPP

#include <Teuchos_ParameterList.hpp>
#include <valarray>

namespace TianXin {

template< typename T >
class GeneralParameter
{
	typedef std::valarray<T>   evalT;
	
	GeneralParameter(const Teuchos::ParameterList& params ) {}
	
	template <class... Args>
	virtual evalT operator()(Args... args) = 0; 

	virtual bool isConstant() const =0;
	virtual unsigned int size() const =0;
};


// **************************************************************
// Constat variables
// **************************************************************

template<typename T>
class ConstantParamter : public GeneralParameter<T>
{
	typedef std::valarray<T>   evalT;
	
    public:
		ConstantParamter(const Teuchos::ParameterList& params );
		evalT operator()(Args... args) final　{return m_value;}
		
		unsigned int size() const final {return m_value.size();}
		bool isConstant() const final   {return true;}

    private:
		evalT m_value;
};


// **************************************************************
// Tableted variables
// **************************************************************

template<typename T>
class TableParamter : public GeneralParameter<T>
{
	typedef std::valarray<T>   evalT;
	
    public:
		TableParamter(const Teuchos::ParameterList& params );
		evalT operator()(Args... args) final;
		
		unsigned int size() const {return m_dependent.size();}
		bool isConstant() const final   {return false;}

    private:
        evalT m_independent;
		std::vector< evalT > m_dependent;
};


}

#include "TianXin_Variables_impl.hpp"

#endif
