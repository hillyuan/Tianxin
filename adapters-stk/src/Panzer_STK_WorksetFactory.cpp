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

#include <Panzer_SetupPartitionedWorksetUtilities.hpp>
#include "Panzer_STK_WorksetFactory.hpp"

#include "Panzer_LocalMeshInfo.hpp"

#include "Panzer_WorksetFactoryBase.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_LocalMeshUtilities.hpp"
#include "Panzer_CommonArrayFactories.hpp"

#include <algorithm>

namespace panzer_stk {

/** Set mesh
  */
void WorksetFactory::setMesh(const Teuchos::RCP<const panzer_stk::STK_Interface> & mesh)
{
   mesh_ = mesh;
}

Teuchos::RCP<std::map<unsigned,panzer::Workset> > WorksetFactory::
getSideWorksets(const panzer::WorksetDescriptor & desc,
                const panzer::WorksetNeeds & needs) const
{
  TEUCHOS_ASSERT(desc.useSideset());

  return panzer_stk::buildBCWorksets(*mesh_,needs,desc.getElementBlock(0),desc.getSideset());
}

Teuchos::RCP<std::map<unsigned,panzer::Workset> > WorksetFactory::
getSideWorksets(const panzer::WorksetDescriptor & desc,
                const panzer::WorksetNeeds & needs_a,
                const panzer::WorksetNeeds & needs_b) const
{
  // ensure that this is a interface descriptor
  TEUCHOS_ASSERT(desc.connectsElementBlocks());
  TEUCHOS_ASSERT(desc.getSideset(0)==desc.getSideset(1));
  return panzer_stk::buildBCWorksets(*mesh_, needs_a, desc.getElementBlock(0),
                                             needs_b, desc.getElementBlock(1),
                                             desc.getSideset(0));
}

Teuchos::RCP<std::vector<panzer::Workset> > WorksetFactory::
getWorksets(const panzer::WorksetDescriptor & worksetDesc,
            const panzer::WorksetNeeds & needs) const
{

  if(worksetDesc.requiresPartitioning()){

    // Generate the local mesh info if it doesn't already exist
    if(mesh_info_ == Teuchos::null){
      TEUCHOS_ASSERT(mesh_ != Teuchos::null);
      mesh_info_ = panzer_stk::generateLocalMeshInfo(*mesh_);
    }

    auto worksets = panzer::buildPartitionedWorksets(*mesh_info_, worksetDesc, this->getOrientationsInterface());

    // Fill in whatever is in the needs object
    // FIXME: This will just get optimized out... Adding volatile to the calls makes the worksets pretty ugly
    for(auto & workset : *worksets){

      // Initialize IntegrationValues from integration descriptors
      for(const auto & id : needs.getIntegrators())
        workset.getIntegrationValues(id);

      // Initialize PointValues from point descriptors
      for(const auto & pd : needs.getPoints())
        workset.getPointValues(pd);

      // Initialize BasisValues
      for(const auto & bd : needs.getBases()){

        // Initialize BasisValues from integrators
        for(const auto & id : needs.getIntegrators())
          workset.getBasisValues(bd,id);

        // Initialize BasisValues from points
        for(const auto & pd : needs.getPoints())
          workset.getBasisValues(bd,pd);
      }
    }

    return worksets;

  } else if(!worksetDesc.useSideset()) {
    // The non-partitioned case always creates worksets with just the
    // owned elements.  CLASSIC_MODE gets the workset size directly
    // from needs that is populated externally. As we transition away
    // from classic mode, we need to create a copy of needs and
    // override the workset size with values from WorksetDescription.
    if (worksetDesc.getWorksetSize() == panzer::WorksetSizeType::CLASSIC_MODE)
      return panzer_stk::buildWorksets(*mesh_,worksetDesc.getElementBlock(), needs);
    else {
      int worksetSize = worksetDesc.getWorksetSize();
      if (worksetSize == panzer::WorksetSizeType::ALL_ELEMENTS) {
        std::vector<stk::mesh::Entity> elements;
        mesh_->getMyElements(worksetDesc.getElementBlock(),elements);
        worksetSize = elements.size();
      }
      panzer::WorksetNeeds tmpNeeds(needs);
	  tmpNeeds.cellData.setCellSize(worksetSize);
      //tmpNeeds.cellData = panzer::CellData(worksetSize,needs.cellData.getCellTopology());
      return panzer_stk::buildWorksets(*mesh_,worksetDesc.getElementBlock(), tmpNeeds);
    }
  }
  else if(worksetDesc.useSideset() && worksetDesc.sideAssembly()) {
    // uses cascade by default, each subcell has its own workset
    return panzer_stk::buildWorksets(*mesh_,needs,worksetDesc.getSideset(),worksetDesc.getElementBlock(),true);
  }
  else {
    TEUCHOS_ASSERT(false);
  }
}

Teuchos::RCP<std::vector<panzer::Workset> >
WorksetFactory :: buildWorksets(const panzer::WorksetDescriptor& worksetDesc,
    const panzer::WorksetNeeds& needs ) const
{
	using LO = panzer::LocalOrdinal;
	panzer::MDFieldArrayFactory mdArrayFactory("",true);
	
	Teuchos::RCP< std::vector<panzer::Workset> > worksets_ptr =
    Teuchos::rcp(new std::vector<panzer::Workset>);
    std::vector<panzer::Workset>& worksets = *worksets_ptr;

	std::vector<panzer::GlobalOrdinal> coords;
	Teuchos::RCP<stk::mesh::MetaData> metaData = mesh_->getMetaData();
    Teuchos::RCP<stk::mesh::BulkData> bulkData = mesh_->getBulkData();
	std::vector<std::size_t> localEntityIds;
	
	const std::string& element_block_name = worksetDesc.getElementBlock();
	Teuchos::RCP<const shards::CellTopology> topo = mesh_->getCellTopology(element_block_name);
	const unsigned n_dim = topo->getDimension();
	const unsigned n_nodes = topo->getNodeCount();
	if(worksetDesc.useSideset()){
	} else {
		int worksetSize = worksetDesc.getWorksetSize();
		const stk::mesh::Part* eb = mesh_->getElementBlockPart(element_block_name);
		if( !eb ) return worksets_ptr;
		std::vector<stk::mesh::Entity> AllElements;
		stk::mesh::Selector eselect = metaData->universal_part() & (*eb);
        stk::mesh::EntityRank elementRank = mesh_->getElementRank();
        stk::mesh::get_selected_entities(eselect,bulkData->buckets(elementRank),AllElements);
		for( const auto& ele : AllElements )
		{
			const auto lid = mesh_->elementLocalId( ele );
			localEntityIds.emplace_back( lid );
		}
		int numElements = localEntityIds.size();

		const int wksize = (worksetSize<=0) ? numElements : std::min( worksetSize, numElements );
		//if(worksetDesc.getWorksetSize() == panzer::WorksetSizeType::ALL_ELEMENTS)
		//	wksize = localEntityIds.size();
		//else
		//	wksize = std::min( worksetSize, localEntityIds.size() );
	    int remain = numElements/wksize;
	    int numWorksets = (numElements-remain)/wksize;
        if( remain>0 ) ++numWorksets;
        worksets.resize(numWorksets);
	
	    // create workset with size = wksize
		LO wkset_count =0;
		LO local_count =0;
		for( LO i=0; i<numElements; i++ )
		{
			worksets[wkset_count].cell_local_ids.emplace_back( localEntityIds[i] );
			++local_count;
			if( local_count>=wksize ) {
				worksets[wkset_count].num_cells = local_count;
				++wkset_count;
				local_count = 0;
			}
		}
		
		wkset_count =0;
		for( auto wkst: worksets )
		{
			wkst.cell_vertex_coordinates = mdArrayFactory.buildStaticArray<double,panzer::Cell,panzer::NODE,panzer::Dim>(
			     "cvc",worksets[wkset_count].num_cells, n_nodes, n_dim);
			wkst.block_id = element_block_name;
			wkst.subcell_dim = needs.cellData.baseCellDimension();
			wkst.subcell_index = 0;
			
			Kokkos::DynRankView<double,PHX::Device> vertex_coordinates;
			mesh_->getElementVertices( wkst.cell_local_ids, vertex_coordinates );
			
			// Copy cell vertex coordinates into local workset arrays
			auto cell_vertex_coordinates = wkst.cell_vertex_coordinates.get_static_view();
			Kokkos::parallel_for(wkst.num_cells, KOKKOS_LAMBDA (int cell) {
			for (std::size_t vertex = 0; vertex < vertex_coordinates.extent(1); ++ vertex)
				for (std::size_t dim = 0; dim < vertex_coordinates.extent(2); ++ dim) {
					cell_vertex_coordinates(cell,vertex,dim) = vertex_coordinates(cell,vertex,dim);
				}
			});
			++wkset_count;
		}
	}
	
	return worksets_ptr;
}

}
