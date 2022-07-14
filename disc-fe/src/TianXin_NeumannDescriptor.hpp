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

#ifndef _TIANXIN_NEUMANN_DESCRIPTOR_HPP
#define _TIANXIN_NEUMANN_DESCRIPTOR_HPP

#include <string>
#include <functional>

/* 
   Descriptor of sideset neumann condition. For convenience to construct NeumannBase evaluator.
   Especially used in response library construction.
*/

namespace TianXin {
	
struct NeumannSidesetDescriptor
{
	std::string ElementBlock;
	std::string SidesetID;
	std::string ResponseName;
	
	NeumannSidesetDescriptor( std::string const& e, std::string const& s, std::string const& r )
	: ElementBlock(e), SidesetID(s), ResponseName(r)
	{}
	
	bool operator == (const NeumannSidesetDescriptor & r)
	{ return ElementBlock==r.ElementBlock && SidesetID==r.SidesetID; }
};
    
}

template<>
struct std::hash<TianXin::NeumannSidesetDescriptor>
{
	std::size_t operator()(TianXin::NeumannSidesetDescriptor const& d) const
	{
		return std::hash<std::string>()(d.ElementBlock + "_" + d.SidesetID);
	}
};

#endif
