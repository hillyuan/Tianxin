// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#ifndef __Panzer_LinearObjContainer_hpp__
#define __Panzer_LinearObjContainer_hpp__

#include "PanzerDiscFE_config.hpp"
#include "Panzer_GlobalEvaluationData.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_dyn_cast.hpp"

namespace panzer {

// **********************************************************************************************
// *********************** LINEAR OBJ CONTAINER *************************************************
// **********************************************************************************************

class LinearObjContainer : public GlobalEvaluationData_Default {
public:
   virtual ~LinearObjContainer() {}

   typedef enum { X=0x1, DxDt=0x2, D2xDt2=0x3, F=0x4, Mat=0x8} Members;

   virtual void initialize() = 0;
   
   virtual void evalDirichletResidual( const std::map< panzer::LocalOrdinal, double >& indx ) = 0;
   virtual void evalDirichletResidual( const Kokkos::View<panzer::LocalOrdinal*, Kokkos::HostSpace>& local_dofs,
		Kokkos::View<double*, Kokkos::HostSpace>& values) = 0;
   virtual void applyDirichletBoundaryCondition( const std::map< panzer::LocalOrdinal, double >& indx ) = 0;
   virtual void applyDirichletBoundaryCondition( const double&, const std::map< panzer::LocalOrdinal, double >& indx ) = 0;
   virtual void applyDirichletBoundaryCondition( const double p, const Kokkos::View<panzer::LocalOrdinal*, Kokkos::HostSpace>& local_dofs,
		Kokkos::View<double*, Kokkos::HostSpace>& values) =0;
   virtual void applyConcentratedLoad( const std::map< panzer::LocalOrdinal, double >& indx ) = 0;
   virtual void applyConcentratedLoad( Kokkos::View<panzer::LocalOrdinal*, Kokkos::HostSpace>& local_dofs,
		Kokkos::View<double*, Kokkos::HostSpace>& values ) = 0;
   
   virtual void writeMatrixMarket(const std::string& filename) const = 0;
};

}

#endif
