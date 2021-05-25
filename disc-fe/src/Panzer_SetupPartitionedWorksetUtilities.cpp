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

#include "Panzer_SetupPartitionedWorksetUtilities.hpp"

#include "Panzer_LocalPartitioningUtilities.hpp"
#include "Panzer_Workset.hpp"
#include "Panzer_WorksetNeeds.hpp"
#include "Panzer_WorksetDescriptor.hpp"

#include <set>
#include <unordered_set>
#include <unordered_map>

namespace panzer
{

namespace
{
void
convertMeshPartitionToWorkset(const panzer::LocalMeshPartition & partition,
                              const Teuchos::RCP<const OrientationsInterface> & orientations,
                              panzer::Workset & workset)
{
  WorksetOptions options;
  options.side_assembly_ = false;
  options.align_side_points_ = false;
  options.orientations_ = orientations;

  // Construct the workset from the partition
  workset.setup(partition, options);

}
	
void
setupSubLocalMeshInfo(const panzer::LocalMeshBlockInfo & parent_info,                 // parent block
                      const std::vector<panzer::LocalOrdinal> & owned_parent_cells,   // cells ID inside this partition
                      panzer::LocalMeshInfoBase & sub_info)                           // output: info for contructing partition
{
  using GO = panzer::GlobalOrdinal;
  using LO = panzer::LocalOrdinal;

  std::size_t n_ghost =0;
  for( auto e: owned_parent_cells )
  	if( e >= parent_info.num_owned_cells ) ++n_ghost;

  std::size_t num_total_cells = owned_parent_cells.size();
  sub_info.num_owned_cells = num_total_cells - n_ghost;
  sub_info.num_ghstd_cells = n_ghost;
  sub_info.num_virtual_cells = 0;

  // Just as a precaution, make sure the parent_info is setup properly
  const int num_parent_total_cells= parent_info.num_owned_cells + parent_info.num_ghstd_cells;
  TEUCHOS_ASSERT(static_cast<int>(parent_info.cell_vertices.extent(0)) == num_parent_total_cells);
  TEUCHOS_ASSERT(static_cast<int>(parent_info.local_cells.extent(0)) == num_parent_total_cells);
  TEUCHOS_ASSERT(static_cast<int>(parent_info.global_cells.extent(0)) == num_parent_total_cells);

  const int num_vertices_per_cell = parent_info.cell_vertices.extent(1);
  const int num_dims = parent_info.cell_vertices.extent(2);

  // Fill owned, ghstd, and virtual cells: global indexes, local indexes and vertices
  sub_info.global_cells = PHX::View<GO*>("global_cells", num_total_cells);
  sub_info.local_cells = PHX::View<LO*>("local_cells", num_total_cells);
  sub_info.cell_vertices = PHX::View<double***>("cell_vertices", num_total_cells, num_vertices_per_cell, num_dims);
  for(int cell=0;cell<num_total_cells;++cell){
    const LO parent_cell = owned_parent_cells[cell];
    sub_info.global_cells(cell) = parent_info.global_cells(parent_cell);
    sub_info.local_cells(cell) = parent_info.local_cells(parent_cell);
    for(int vertex=0;vertex<num_vertices_per_cell;++vertex){
      for(int dim=0;dim<num_dims;++dim){
        sub_info.cell_vertices(cell,vertex,dim) = parent_info.cell_vertices(parent_cell,vertex,dim);
      }
    }
  }

  // Face part
  TEUCHOS_ASSERT(static_cast<int>(parent_info.cell_to_faces.extent(0)) == num_parent_total_cells);
  const int num_faces_per_cell = parent_info.cell_to_faces.extent(1);
  sub_info.cell_to_faces = PHX::View<LO**>("cell_to_faces", num_total_cells, num_faces_per_cell);
  std::set<LO> faceLIDs;
  for(int cell=0;cell<num_total_cells;++cell){
    const LO parent_cell = owned_parent_cells[cell];
	for( int face=0; face<num_faces_per_cell; ++ face) {
    	sub_info.cell_to_faces(cell, face) = parent_info.cell_to_faces(parent_cell, face);
		faceLIDs.insert( parent_info.cell_to_faces(parent_cell, face) );
    }
  }
  
  const int num_faces = faceLIDs.size();
  sub_info.face_to_cells = PHX::View<LO*[2]>("face_to_cells", num_faces);
  sub_info.face_to_lidx = PHX::View<LO*[2]>("face_to_lidx", num_faces);
  std::size_t nf =0;
  for( const LO face_LID : faceLIDs) {
	sub_info.face_to_cells(nf,0) = parent_info.face_to_cells(face_LID,0);
	sub_info.face_to_cells(nf,1) = parent_info.face_to_cells(face_LID,1);
	sub_info.face_to_lidx(nf,0) = parent_info.face_to_lidx(face_LID,0);
	sub_info.face_to_lidx(nf,1) = parent_info.face_to_lidx(face_LID,1);
//	std::cout << "face:" << face_LID << "," << sub_info.face_to_lidx(nf,0) << "," << sub_info.face_to_lidx(nf,1) <<std::endl;
	nf++;
  }

}

	
void
splitMeshInfo(const panzer::LocalMeshBlockInfo & block_info,
              const int splitting_size,
              std::vector<panzer::LocalMeshPartition> & partitions)
{

  using LO = panzer::LocalOrdinal;

  // Make sure the splitting size makes sense
  TEUCHOS_ASSERT((splitting_size > 0) or (splitting_size == WorksetSizeType::ALL_ELEMENTS));

  // Default partition size
  const LO numCells = block_info.num_owned_cells + block_info.num_ghstd_cells;
  const LO base_partition_size = std::min(numCells, (splitting_size > 0) ? splitting_size : numCells);

  // Cells to partition
  std::vector<LO> partition_cells;
  partition_cells.resize(base_partition_size);

  // Create the partitions
  LO cell_count = 0;
  while(cell_count < numCells){

    LO partition_size = base_partition_size;
    if(cell_count + partition_size > numCells)
      partition_size = numCells - cell_count;

    // Error check for a null partition - this should never happen by design
    TEUCHOS_ASSERT(partition_size != 0);

    // In the final partition, we need to reduce the size of partition_cells
    if(partition_size != base_partition_size)
      partition_cells.resize(partition_size);

    // Set the partition indexes - not really a partition, just a chunk of cells
    for(LO i=0; i<partition_size; ++i)
      partition_cells[i] = cell_count+i;

    // Create an empty partition
    partitions.push_back(panzer::LocalMeshPartition());

    // Fill the empty partition
    setupSubLocalMeshInfo(block_info,partition_cells,partitions.back());

    // Update the cell count
    cell_count += partition_size;
  }

}
	
void
generateLocalMeshPartitions_new(const panzer::LocalMeshInfo & mesh_info,
                            const panzer::WorksetDescriptor & description,
                            std::vector<panzer::LocalMeshPartition> & partitions)
{
  // We have to make sure that the partitioning is possible
  TEUCHOS_ASSERT(description.getWorksetSize() != panzer::WorksetSizeType::CLASSIC_MODE);
  TEUCHOS_ASSERT(description.getWorksetSize() != 0);

  // This could just return, but it would be difficult to debug why no partitions were returned
  TEUCHOS_ASSERT(description.requiresPartitioning());

  const std::string & element_block_name = description.getElementBlock();

  // We have two processes for in case this is a sideset or element block
  if(description.useSideset()){

    // If the element block doesn't exist, there are no partitions to create
    if(mesh_info.sidesets.find(element_block_name) == mesh_info.sidesets.end())
      return;
    const auto & sideset_map = mesh_info.sidesets.at(element_block_name);

    const std::string & sideset_name = description.getSideset();

    // If the sideset doesn't exist, there are no partitions to create
    if(sideset_map.find(sideset_name) == sideset_map.end())
      return;

    const panzer::LocalMeshSidesetInfo & sideset_info = sideset_map.at(sideset_name);

    // Partitioning is not important for sidesets
//    panzer::partitioning_utilities::splitMeshInfo(sideset_info, description.getWorksetSize(), partitions);

    for(auto & partition : partitions){
      partition.sideset_name = sideset_name;
      partition.element_block_name = element_block_name;
      partition.cell_topology = sideset_info.cell_topology;
      partition.has_connectivity = true;
    }

  } else {

    // If the element block doesn't exist, there are no partitions to create
    if(mesh_info.element_blocks.find(element_block_name) == mesh_info.element_blocks.end())
      return;

    // Grab the element block we're interested in
    const panzer::LocalMeshBlockInfo & block_info = mesh_info.element_blocks.at(element_block_name);

    if(description.getWorksetSize() == panzer::WorksetSizeType::ALL_ELEMENTS){
      // We only have one partition describing the entire local mesh		d::cout ｓ
      splitMeshInfo(block_info, -1, partitions);
    } else {
      // We need to partition local mesh

      // FIXME: Until the above function is fixed, we will use this hack - this will lead to horrible partitions
      splitMeshInfo(block_info, description.getWorksetSize(), partitions);

    }

    for(auto & partition : partitions){
      partition.element_block_name = element_block_name;
      partition.cell_topology = block_info.cell_topology;
      partition.has_connectivity = true;
    }
  }

}

}


Teuchos::RCP<std::vector<panzer::Workset> >  
buildPartitionedWorksets(const panzer::LocalMeshInfo & mesh_info,
                         const panzer::WorksetDescriptor & description,
                         const Teuchos::RCP<const OrientationsInterface> & orientations)
{
  Teuchos::RCP<std::vector<panzer::Workset> > worksets = Teuchos::rcp(new std::vector<panzer::Workset>());

  // Make sure it makes sense to partition
  TEUCHOS_ASSERT(description.requiresPartitioning());

  // Each partition represents a chunk of the mesh
  std::vector<panzer::LocalMeshPartition> partitions;
  generateLocalMeshPartitions_new(mesh_info, description, partitions);

  int i=0;
  for(const auto & partition : partitions){
    worksets->push_back(panzer::Workset());
    convertMeshPartitionToWorkset(partition, orientations, worksets->back());

    // We hash in a unique id the the given workset
    size_t id = std::hash<WorksetDescriptor>()(description);
    panzer::hash_combine(id, i++);
    worksets->back().setIdentifier(id);
  }

  return worksets;
}

}
