// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
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
//  Copyright (2022) YUAN Xi
// ******************************************************************* 
// @HEADER

#include "Panzer_CellTopologyInfo.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

panzer::CellTopologyInfo::
CellTopologyInfo(int numCells, const shards::CellTopology& cellTopo)
	: num_cells(numCells), topology(cellTopo)
{
  int num_edges = cellTopo.getEdgeCount();
  int dimension = topology.getDimension(); 
  edge_scalar = Teuchos::rcp(new PHX::MDALayout<Cell,Edge>(numCells, num_edges));
  edge_vector = Teuchos::rcp(new PHX::MDALayout<Cell,Edge,Dim>(numCells, num_edges, dimension));
}


panzer::CellTopologyInfo& 
panzer::CellTopologyInfo::operator=(const panzer::CellTopologyInfo& topo)
{
  this->num_cells = topo.num_cells;
  this->topology = topo.topology;
  int num_edges = topo.getNumEdges();
  int dimension = topo.getDimension(); 
  edge_scalar = Teuchos::rcp(new PHX::MDALayout<Cell,Edge>(num_cells, num_edges));
  edge_vector = Teuchos::rcp(new PHX::MDALayout<Cell,Edge,Dim>(num_cells, num_edges, dimension));
	return *this;
}

