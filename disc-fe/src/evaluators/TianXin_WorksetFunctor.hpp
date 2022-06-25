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

#ifndef _TIANXIN_WORKSET_FUNCTOR_HPP
#define _TIANXIN_WORKSET_FUNCTOR_HPP

#include "Panzer_Workset.hpp"
#include <Teuchos_ParameterList.hpp>
#include "TianXin_Factory.hpp"

namespace TianXin {

class WorksetFunctor
{
  public:
    WorksetFunctor(const Teuchos::ParameterList& params ) {}
    virtual double operator()(const panzer::Workset&) = 0;
};

typedef Factory<WorksetFunctor,std::string,Teuchos::ParameterList> WorksetFunctorFactory;

// **************************************************************
// Constat function
// **************************************************************

template<typename EvalT>
class ConstantWorksetFunctor : public WorksetFunctor
{
  public:
    ConstantWorksetFunctor(const Teuchos::ParameterList& params );
    double operator()(const panzer::Workset&) final;
  private:
    double m_value;
};
namespace FunctorRegister {
	 static bool const ok = WorksetFunctorFactory::Instance().template Register< ConstantWorksetFunctor<double> >( "Constant");
}
// **************************************************************
// Linear function
// **************************************************************

template<typename EvalT>
class LinearWorksetFunctor : public WorksetFunctor
{
  public:
    LinearWorksetFunctor(const Teuchos::ParameterList& params );
    double operator()(const panzer::Workset&) final;
  private:
    double m_elapse_time;
    double m_value;
};
namespace FunctorRegister {
	static bool const ok1 = WorksetFunctorFactory::Instance().template Register< LinearWorksetFunctor<double> >( "Linear");
}

// **************************************************************
// Table Function of Time
// **************************************************************

template<typename EvalT>
class TimeTableFunctor : public WorksetFunctor
{
  public:
    TimeTableFunctor(const Teuchos::ParameterList& params );
    double operator()(const panzer::Workset&) final;
  private:
    std::vector<double> m_time;
    std::vector<double> m_value;
};
namespace FunctorRegister {
	static bool const ok2 = WorksetFunctorFactory::Instance().template Register< TimeTableFunctor<double> >( "TimeTable");
}

// **************************************************************
// Function of time expression
// **************************************************************

template<typename EvalT>
class TimeExpressionFunctor : public WorksetFunctor
{
  public:
    TimeExpressionFunctor(const Teuchos::ParameterList& params );
    double operator()(const panzer::Workset&) final;
  private:
    std::string expression{""};
};

// **************************************************************
// Function of coordinate expression
// **************************************************************

/*template<typename EvalT>
class CoordExpressionFunctor : public WorksetFunctor<EvalT>
{
  public:
    CoordExpressionFunctor(const Teuchos::ParameterList& params );
    ScalarT operator(panzer::workset&) final;
  private:
    std::string expression{""};
};*/

/*template<typename EvalT>
WorksetFunctor* createConstantFunctor(const Teuchos::ParameterList& params)             
{ 
    return new ConstantFunctor<EvalT>(params);     
}


template<typename EvalT>
WorksetFunctor* createLinearFunctor(const Teuchos::ParameterList& params)             
{ 
    return new LinearFunctor<EvalT>(params);     
}
bool const ok1 = WorksetFunctorFactory::Instance().Register( "Linear", createLinearFunctor<double>);*/


}

#include "TianXin_WorksetFunctor_impl.hpp"

#endif
