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

#include "Panzer_STK_WorksetFactory.hpp"

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

   if(!worksetDesc.useSideset()) {
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
      tmpNeeds.cellData = panzer::CellData(worksetSize,needs.cellData.getCellTopology());
      return panzer_stk::buildWorksets(*mesh_,worksetDesc.getElementBlock(), needs);
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

Teuchos::RCP<std::vector<panzer::Workset> > WorksetFactory::
WorksetFactory :: generateWorksets(const panzer::WorksetDescriptor& worksetDesc,
    const panzer::WorksetNeeds& needs ) const
{
	using LO = panzer::LocalOrdinal;
	panzer::MDFieldArrayFactory mdArrayFactory("",true);
	
	Teuchos::RCP<std::vector<panzer::Workset> > worksets_ptr = Teuchos::rcp(new std::vector<panzer::Workset>);
	int myrank = mesh_->getComm()->getRank();
	
//	std::vector<panzer::GlobalOrdinal> coords;
//	Teuchos::RCP<stk::mesh::MetaData> metaData = mesh_->getMetaData();
//    Teuchos::RCP<stk::mesh::BulkData> bulkData = mesh_->getBulkData();
	std::vector<panzer::LocalOrdinal> local_cell_ids;
	std::vector<stk::mesh::Entity> elements;
	
	const std::string& element_block_name = worksetDesc.getElementBlock();
	Teuchos::RCP<const shards::CellTopology> topo = mesh_->getCellTopology(element_block_name);
	const int n_dim = topo->getDimension();
	const int n_nodes = topo->getNodeCount();
	const int n_celldata = needs.cellData.numCells();
	const int n_sides = topo->getSubcellCount(n_dim-1);
	
/*	if(worksetDesc.useSideset()){
		std::vector<stk::mesh::Entity> sideEntities;
		mesh_->getMySides(worksetDesc.getSideset(),element_block_name,sideEntities);
		if( sideEntities.empty() ) return;   // no entity in current cpu
		std::vector<std::size_t> local_side_ids;
		panzer_stk::workset_utils::getSideElements(*mesh_, element_block_name,sideEntities,local_side_ids,elements);
		for(const auto& element: elements) {
			local_cell_ids.emplace_back(mesh_->elementLocalId(element));
		}
	} else*/
	{
		int worksetSize = worksetDesc.getWorksetSize();
		if( worksetSize>0 ) {
			if( n_celldata>0 && n_celldata<worksetSize ) worksetSize = n_celldata;
		} else {
			worksetSize = n_celldata;
		}
		//const stk::mesh::Part* eb = mesh_->getElementBlockPart(element_block_name);
		//if( !eb ) return;
		//stk::mesh::Selector eselect = metaData->universal_part() & (*eb);
        //stk::mesh::EntityRank elementRank = mesh_->getElementRank();
        //stk::mesh::get_selected_entities(eselect,bulkData->buckets(elementRank),elements);
		mesh_->getMyElements(element_block_name,elements);
		if( elements.empty() ) return worksets_ptr;     // no entity in current cpu
		for( const auto& ele : elements )
		{
			const auto lid = mesh_->elementLocalId( ele );
			local_cell_ids.emplace_back( lid );
		}		 
		int numElements = local_cell_ids.size();
		if( worksetDesc.getWorksetSize() == panzer::WorksetSizeType::ALL_ELEMENTS ) 
			worksetSize = -1;

		const int wksize = (worksetSize<=0) ?  numElements : std::min( worksetSize, numElements );
		//if(worksetDesc.getWorksetSize() == panzer::WorksetSizeType::ALL_ELEMENTS)
		//	wksize = local_cell_ids.size();
		//else
		//	wksize = std::min( worksetSize, local_cell_ids.size() );
	    int remain = numElements%wksize;
	    int numWorksets = (numElements-remain)/wksize;
        if( remain>0 ) ++numWorksets;
        worksets_ptr->resize(numWorksets);

	    // create workset with size = wksize
		LO wkset_count =0;
		LO local_count =0;
		for( LO i=0; i<numElements; i++ )
		{
			worksets_ptr->at(wkset_count).cell_local_ids.emplace_back( local_cell_ids[i] );
			++local_count;
			if( local_count>=wksize ) {
				worksets_ptr->at(wkset_count).num_cells = local_count;
				worksets_ptr->at(wkset_count).setNumeberOwnedCells(local_count);
				++wkset_count;
				local_count = 0;
			}
		}

		std::vector<panzer::LocalOrdinal> side2ele, ele2side;
		for( LO i=0; i<numWorksets; i++ )
		{
			std::size_t n_ele = worksets_ptr->at(i).num_cells;
			worksets_ptr->at(i).cell_vertex_coordinates = mdArrayFactory.buildStaticArray<double,panzer::Cell,panzer::NODE,panzer::Dim>(
			     "cvc", n_ele, n_nodes, n_dim);		
			worksets_ptr->at(i).block_id = element_block_name;
			worksets_ptr->at(i).set_dimension( n_dim );
			worksets_ptr->at(i).subcell_dim = n_dim-1;
			worksets_ptr->at(i).subcell_index = 0;
			worksets_ptr->at(i).setTopology(topo);

            PHX::View<int*> cell_local_ids_k = PHX::View<int*>("Workset:cell_local_ids",n_ele);
			auto cell_local_ids_k_h = Kokkos::create_mirror_view(cell_local_ids_k);
			for(std::size_t j=0;j<n_ele;j++)
				cell_local_ids_k_h(j) = worksets_ptr->at(i).cell_local_ids[j];
			Kokkos::deep_copy(cell_local_ids_k, cell_local_ids_k_h);
			worksets_ptr->at(i).cell_local_ids_k = cell_local_ids_k;

			Kokkos::DynRankView<double,PHX::Device> vertex_coordinates;
			mesh_->getElementVertices( worksets_ptr->at(i).cell_local_ids, vertex_coordinates );
			
			// Copy cell vertex coordinates into local workset arrays
			auto cell_vertex_coordinates = worksets_ptr->at(i).cell_vertex_coordinates.get_static_view();
			Kokkos::parallel_for(worksets_ptr->at(i).num_cells, KOKKOS_LAMBDA (int cell) {
			for (std::size_t vertex = 0; vertex < vertex_coordinates.extent(1); ++ vertex)
				for (std::size_t dim = 0; dim < vertex_coordinates.extent(2); ++ dim) {
					cell_vertex_coordinates(cell,vertex,dim) = vertex_coordinates(cell,vertex,dim);
				}
			});
			
			worksets_ptr->at(i).setSetup(true);
			mesh_->getElementSideRelation( worksets_ptr->at(i).cell_local_ids,side2ele, ele2side);
			worksets_ptr->at(i).setupFaceConnectivity(side2ele, n_sides, ele2side);

			// Initialize IntegrationValues from integration descriptors
			const auto& integs = needs.getIntegrators();
			for(const auto & id : integs)
				worksets_ptr->at(i).getIntegrationValues(id);

			// Initialize PointValues from point descriptors
			const auto& points = needs.getPoints();
			for(const auto & pd : points)
				worksets_ptr->at(i).getPointValues(pd);

			// Initialize BasisValues
			const auto& basis = needs.getBases();
			if( basis.size()>0 ) {
				TEUCHOS_ASSERT( needs.bases.size()==0 );  // no legacy definition
				for(const auto & bd : basis){

					// Initialize BasisValues from integrators
					for(const auto & id : needs.getIntegrators())
						worksets_ptr->at(i).getBasisValues(bd,id);

					// Initialize BasisValues from points
					for(const auto & pd : needs.getPoints())
					worksets_ptr->at(i).getBasisValues(bd,pd);
				} 
			}
		//	else if (needs.bases.size()>0) {
		//		for( const auto& nds : needs.bases) {
		//			panzer::BasisDescriptor bd(nds->order(),nds->type());
		//		}
		//	}
		//	std::cout <<  worksets_ptr->at(i) ;
		}
	}
	return worksets_ptr;
}

Teuchos::RCP<std::vector<panzer::Workset> > WorksetFactory::
WorksetFactory :: generateWorksets(const panzer::PhysicsBlock& pb ) const
{
	using LO = panzer::LocalOrdinal;
	panzer::MDFieldArrayFactory mdArrayFactory("",true);
	
	Teuchos::RCP<std::vector<panzer::Workset> > pwksets = Teuchos::rcp(new std::vector<panzer::Workset>);
	std::vector<panzer::LocalOrdinal> local_cell_ids;
	std::vector<stk::mesh::Entity> elements;
	
	const auto& eb = pb.elementBlockID();
	const auto& needs = pb.getWorksetNeeds();
	const int worksetSize = needs.cellData.numCells();  // need reading?
	
	Teuchos::RCP<const shards::CellTopology> topo = mesh_->getCellTopology(eb);
	const int n_dim = topo->getDimension();
	const int n_nodes = topo->getNodeCount();
	const int n_celldata = needs.cellData.numCells();
	const int n_sides = topo->getSubcellCount(n_dim-1);
	
	mesh_->getMyElements(eb,elements);
	if( elements.empty() ) return pwksets;     // no entity in current cpu
	for( const auto& ele : elements )
	{
		const auto lid = mesh_->elementLocalId( ele );
		local_cell_ids.emplace_back( lid );
	}		 
	int numElements = local_cell_ids.size();
	const int wksize = std::min( worksetSize, numElements );
	
	int remain = numElements%wksize;
	int numWorksets = (numElements-remain)/wksize;
    if( remain>0 ) ++numWorksets;
    pwksets->resize(numWorksets);

	// create workset with size = wksize
	LO wkset_count =0;
	LO local_count =0;
	for( LO i=0; i<numElements; i++ )
	{
		pwksets->at(wkset_count).cell_local_ids.emplace_back( local_cell_ids[i] );
		++local_count;
		if( local_count>=wksize ) {
			pwksets->at(wkset_count).num_cells = local_count;
			pwksets->at(wkset_count).setNumeberOwnedCells(local_count);
			++wkset_count;
			local_count = 0;
		}
	}

    std::vector<panzer::LocalOrdinal> side2ele, ele2side;
	for( LO i=0; i<numWorksets; i++ )
	{
		std::size_t n_ele = pwksets->at(i).num_cells;
		pwksets->at(i).cell_vertex_coordinates = mdArrayFactory.buildStaticArray<double,panzer::Cell,panzer::NODE,panzer::Dim>(
			     "cvc", n_ele, n_nodes, n_dim);		
		pwksets->at(i).block_id = eb;
		pwksets->at(i).set_dimension( n_dim );
		pwksets->at(i).subcell_dim = n_dim-1;
		pwksets->at(i).subcell_index = 0;
		pwksets->at(i).setTopology(topo);

        PHX::View<int*> cell_local_ids_k = PHX::View<int*>("Workset:cell_local_ids",n_ele);
		auto cell_local_ids_k_h = Kokkos::create_mirror_view(cell_local_ids_k);
		for(std::size_t j=0;j<n_ele;j++)
			cell_local_ids_k_h(j) = pwksets->at(i).cell_local_ids[j];
		Kokkos::deep_copy(cell_local_ids_k, cell_local_ids_k_h);
		pwksets->at(i).cell_local_ids_k = cell_local_ids_k;

		Kokkos::DynRankView<double,PHX::Device> vertex_coordinates;
		mesh_->getElementVertices( pwksets->at(i).cell_local_ids, vertex_coordinates );
			
		// Copy cell vertex coordinates into local workset arrays
		auto cell_vertex_coordinates = pwksets->at(i).cell_vertex_coordinates.get_static_view();
		Kokkos::parallel_for(pwksets->at(i).num_cells, KOKKOS_LAMBDA (int cell) {
		for (std::size_t vertex = 0; vertex < vertex_coordinates.extent(1); ++ vertex)
			for (std::size_t dim = 0; dim < vertex_coordinates.extent(2); ++ dim) {
				cell_vertex_coordinates(cell,vertex,dim) = vertex_coordinates(cell,vertex,dim);
			}
		});
			
		pwksets->at(i).setSetup(true);
		mesh_->getElementSideRelation( pwksets->at(i).cell_local_ids,side2ele, ele2side);
		pwksets->at(i).setupFaceConnectivity(side2ele, n_sides, ele2side);
		// Initialize IntegrationValues from integration descriptors
		const auto& integs = needs.getIntegrators();
		for(const auto & id : integs)
			pwksets->at(i).getIntegrationValues(id);

		// Initialize PointValues from point descriptors
		const auto& points = needs.getPoints();
		for(const auto & pd : points)
			pwksets->at(i).getPointValues(pd);

		// Initialize BasisValues
		const auto& basis = needs.getBases();
		for(const auto & bd : basis){

			// Initialize BasisValues from integrators
			for(const auto & id : needs.getIntegrators())
				pwksets->at(i).getBasisValues(bd,id);

			// Initialize BasisValues from points
			for(const auto & pd : needs.getPoints())
			pwksets->at(i).getBasisValues(bd,pd);
		}
		//	std::cout <<  pwksets->at(i) ;
	}
	
	return pwksets;
}

}
