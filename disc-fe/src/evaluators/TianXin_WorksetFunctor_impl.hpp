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

#ifndef _TIANXIN_WORKSET_FUNCTOR_IMPL_HPP
#define _TIANXIN_WORKSET_FUNCTOR_IMPL_HPP

//#include <stk_expreval/Evaluator.hpp>

namespace TianXin {

// **************************************************************
// ConstantFunctor
// **************************************************************
template<typename EvalT>
ConstantWorksetFunctor<EvalT>::ConstantWorksetFunctor(const Teuchos::ParameterList& params )
: WorksetFunctor(params)
{
	const Teuchos::ParameterList& p = params.sublist("Constant"); 
	m_value = p.get<double>("Value");
}


template<typename EvalT>
double ConstantWorksetFunctor<EvalT> :: operator()(const panzer::Workset&)
{
    return m_value;
}

// **************************************************************
// LinearFunctor
// **************************************************************
template<typename EvalT>
LinearWorksetFunctor<EvalT>::LinearWorksetFunctor(const Teuchos::ParameterList& params )
: WorksetFunctor(params)
{
	const Teuchos::ParameterList& p = params.sublist("Constant"); 
	m_value = p.get<double>("Value");
	m_elapse_time = p.get<double>("Value");
}


template<typename EvalT>
double LinearWorksetFunctor<EvalT> :: operator()(const panzer::Workset& wk)
{
	if( wk.time>= m_elapse_time ) return m_value;
	if( wk.time<= 0.0 ) return 0.0;
    return m_value*wk.time/m_elapse_time;
}

// **************************************************************
// TimeTableFunctor
// **************************************************************
template<typename EvalT>
TimeTableFunctor<EvalT>::TimeTableFunctor(const Teuchos::ParameterList& params )
: WorksetFunctor(params)
{
	const Teuchos::ParameterList& p = params.sublist("TimeTable"); 
	m_time = p.get<Teuchos::Array<double>>("Time Values").toVector();
    m_value   = p.get<Teuchos::Array<double>>("BC Values").toVector();
}


template<typename EvalT>
double TimeTableFunctor<EvalT> :: operator()(const panzer::Workset& wk)
{
	double current_time = wk.time;
    TEUCHOS_TEST_FOR_EXCEPTION(
      current_time > m_time.back(),
      Teuchos::Exceptions::InvalidParameter,
      "Time is growing unbounded!");

    double      val;
    double      slope;
    unsigned int index(0);

    while (m_time[index] < current_time) index++;

    if (index == 0)
      val = m_value[index];
    else {
      slope = ((m_value[index] - m_value[index - 1]) /
         (m_time[index] - m_time[index - 1]));
      val = m_value[index - 1] + slope * (current_time - m_time[index - 1]);
    }

    return val;
}

// **************************************************************
// TimeExpressionFunctor
// **************************************************************
template<typename EvalT>
TimeExpressionFunctor<EvalT>::TimeExpressionFunctor(const Teuchos::ParameterList& params )
: WorksetFunctor(params)
{
	const Teuchos::ParameterList& p = params.sublist("Constant"); 
	expression = p.get<std::string>("Expression");
}

template<typename EvalT>
double TimeExpressionFunctor<EvalT> :: operator()(const panzer::Workset& wk)
{
/*	stk::expreval::Eval expr_eval(expression);
    expr_eval.parse();
	double curtime = wk.time;
    expr_eval.bindVariable("t", curtime);
    return expr_eval.evaluate();*/
	return 0.0;
}

}

#endif
