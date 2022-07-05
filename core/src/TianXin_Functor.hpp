// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) YUAN Xi
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

#ifndef _TIANXIN_FUNCTOR_HPP
#define _TIANXIN_FUNCTOR_HPP

#include <Teuchos_ParameterList.hpp>
#include "TianXin_Factory.hpp"
#include <vector>

namespace TianXin {

template< typename T >
struct GeneralFunctor
{
	GeneralFunctor(const Teuchos::ParameterList& params ) {}
	
	virtual std::vector<T> operator()(std::initializer_list<T>) const = 0; 
	virtual std::vector<T> operator()() const
	{
		return this->operator()(std::initializer_list<T>({}));
	}

	virtual bool isConstant() const =0;
	virtual unsigned int nitems() const =0;
};

typedef Factory<GeneralFunctor<double>,std::string,Teuchos::ParameterList> GeneralFunctorFactory;

// **************************************************************
// Constant variables
// **************************************************************

template< typename T >
class ConstantFunctor : public GeneralFunctor<T>
{
    public:
		ConstantFunctor(const Teuchos::ParameterList& params );
		std::vector<T> operator()(std::initializer_list<T>) const final {return m_value;}
		
		unsigned int nitems() const final {return m_value.size();}
		bool isConstant() const final   {return true;}

    private:
		std::vector<T> m_value;
};

namespace GeneralFunctorRegister {
	 static bool const Constant_OK = GeneralFunctorFactory::Instance().template Register< ConstantFunctor<double> >( "Constant");
}
// **************************************************************
// Table variables
// **************************************************************

/*template< typename T >
class TableParamter : public GeneralParameter<T>
{
    public:
		TableParamter(const Teuchos::ParameterList& params );
		std::vector<T> operator()(std::initializer_list<T>) const final;
		
		unsigned int nitems() const final {return m_dependent.size();}
		bool isConstant() const final   {return false;}

    private:
        std::valarray<T> m_independent;
		std::vector< T > m_dependent;
};*/


}

#include "TianXin_Functor_impl.hpp"

#endif
