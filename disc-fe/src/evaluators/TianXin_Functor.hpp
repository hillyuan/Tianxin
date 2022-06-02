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

#ifndef _WORKSET_FUNCTOR_HPP
#define _WORKSET_FUNCTOR_HPP

#include "Panzer_Workset.hpp"
#include <Teuchos_ParameterList.hpp>

#include <cassert>

namespace TianXin {

template<typename EvalT>
class WorksetFunctor
{
  protected:
    typedef typename EvalT::ScalarT ScalarT;

  public:
    WorksetFunctor(const Teuchos::ParameterList& params ) {}
    virtual double operator()(const panzer::Workset&) = 0;
};

// **************************************************************
// Constat function
// **************************************************************

template<typename EvalT>
class ConstantFunctor : public WorksetFunctor<EvalT>
{
  using typename WorksetFunctor<EvalT>::ScalarT;

  public:
    ConstantFunctor(const Teuchos::ParameterList& params );
    double operator()(const panzer::Workset&) final;
  private:
    double m_value;
};


// **************************************************************
// Linear function
// **************************************************************

template<typename EvalT>
class LinearFunctor : public WorksetFunctor<EvalT>
{
  using typename WorksetFunctor<EvalT>::ScalarT;

  public:
    LinearFunctor(const Teuchos::ParameterList& params );
    double operator()(const panzer::Workset&) final;
  private:
    double m_elapse_time;
    double m_value;
};


// **************************************************************
// Table Function of Time
// **************************************************************

template<typename EvalT>
class TimeTableFunctor : public WorksetFunctor<EvalT>
{
  using typename WorksetFunctor<EvalT>::ScalarT;

  public:
    TimeTableFunctor(const Teuchos::ParameterList& params );
    double operator()(const panzer::Workset&) final;
  private:
    std::vector<double> m_time;
    std::vector<double> m_value;
};

// **************************************************************
// Function of time expression
// **************************************************************

template<typename EvalT>
class TimeExpressionFunctor : public WorksetFunctor<EvalT>
{
  using typename WorksetFunctor<EvalT>::ScalarT;

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


}

#include "TianXin_Functor_impl.hpp"

#endif
